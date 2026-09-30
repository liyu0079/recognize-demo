"""Dedicated Moondream2 language decoder exporter with Stateful KV cache."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class DecoderExportError(RuntimeError):
    pass


def _decoder_wrapper(native: Any) -> Any:
    """Create a stateless PyTorch graph with explicit 24-layer KV tensors.

    OpenVINO's state transformation needs a conventional input/output pair.
    Moondream's native ``KVCache.update`` mutates Python buffers, so exporting
    it directly loses the visual prefix. This graph keeps exactly the same
    weight layout while returning the concatenated cache explicitly.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from _moondream2_openvino_source.rope import apply_rotary_emb

    text = native.model.text
    config = native.model.config.text

    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text = text
            self.n_heads = config.n_heads
            self.n_kv_heads = config.n_kv_heads
            self.head_dim = config.dim // config.n_heads

        @staticmethod
        def _layer_norm(value: torch.Tensor, layer: nn.Module) -> torch.Tensor:
            return F.layer_norm(value, layer.bias.shape, layer.weight, layer.bias)

        @staticmethod
        def _mlp(value: torch.Tensor, block: nn.Module) -> torch.Tensor:
            value = F.linear(value, block["fc1"].weight, block["fc1"].bias)
            value = F.gelu(value, approximate="tanh")
            return F.linear(value, block["fc2"].weight, block["fc2"].bias)

        def forward(
            self,
            inputs_embeds: torch.Tensor,
            position_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            past_key_values: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # past_key_values: [24, 2, B, 32, past_length, 64]
            hidden = inputs_embeds
            present: list[torch.Tensor] = []
            bsz, tokens, _ = hidden.shape
            for index, block in enumerate(self.text["blocks"]):
                normalized = self._layer_norm(hidden, block["ln"])
                qkv = F.linear(normalized, block["attn"]["qkv"].weight, block["attn"]["qkv"].bias)
                q_size = self.n_heads * self.head_dim
                kv_size = self.n_kv_heads * self.head_dim
                q, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
                q = q.view(bsz, tokens, self.n_heads, self.head_dim).transpose(1, 2)
                key = key.view(bsz, tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)
                value = value.view(bsz, tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)
                q = apply_rotary_emb(q, self.text.freqs_cis, position_ids, self.n_heads)
                key = apply_rotary_emb(key, self.text.freqs_cis, position_ids, self.n_kv_heads)
                full_key = torch.cat([past_key_values[index, 0], key], dim=2)
                full_value = torch.cat([past_key_values[index, 1], value], dim=2)
                attended = F.scaled_dot_product_attention(q, full_key, full_value, attn_mask=attention_mask)
                attended = attended.transpose(1, 2).reshape(bsz, tokens, -1)
                attended = F.linear(attended, block["attn"]["proj"].weight, block["attn"]["proj"].bias)
                hidden = hidden + attended + self._mlp(normalized, block["mlp"])
                present.append(torch.stack([full_key, full_value], dim=0))
            normalized = F.layer_norm(hidden[:, -1, :], self.text["post_ln"].bias.shape, self.text["post_ln"].weight, self.text["post_ln"].bias)
            logits = F.linear(normalized, self.text["lm_head"].weight, self.text["lm_head"].bias)
            return logits, torch.stack(present, dim=0)

    return Decoder().eval()


def _make_stateful(model: Any) -> None:
    """Pair past/present cache ports using OpenVINO ReadValue/Assign passes."""
    try:
        from openvino._offline_transformations import apply_make_stateful_transformation

        apply_make_stateful_transformation(model, {"past_key_values": "present_key_values"})
    except Exception as exc:
        raise DecoderExportError(
            "Unable to make decoder cache stateful. Ensure OpenVINO 2024.3+ is installed. "
            f"Details: {exc}",
        ) from exc


def export_decoder(
    native: Any,
    output_dir: Path,
    max_context: int = 2048,
    precision: str = "fp16",
) -> Path:
    """Export ``decoder.xml`` plus state metadata consumed by the adapter."""
    import openvino as ov
    import torch

    if precision not in {"fp16", "int8"}:
        raise DecoderExportError("precision must be fp16 or int8")
    if precision == "int8":
        # INT8 must be calibrated against representative local caption/image
        # prompts. Do not leave a misleading FP16 file in an INT8 directory.
        raise DecoderExportError("INT8 requested, but no local calibration dataset was supplied; refusing lossy uncalibrated export")
    config = native.model.config.text
    if max_context < 730 or max_context > config.max_context:
        raise DecoderExportError(f"max_context must be in [730, {config.max_context}]")
    output_dir.mkdir(parents=True, exist_ok=True)
    decoder = _decoder_wrapper(native)
    # Empty past caches are not accepted by all Torch export frontends. One
    # visual-prefix token establishes a dynamic cache dimension; OpenVINO is
    # reshaped to allow 0..max_context afterwards and reset_state clears it.
    example = (
        torch.zeros((1, 1, config.dim), dtype=native.model.vision.pos_emb.dtype),
        torch.zeros((1,), dtype=torch.long),
        torch.ones((1, 1, 1, 2), dtype=torch.bool),
        torch.zeros((config.n_layers, 2, 1, config.n_kv_heads, 1, config.dim // config.n_heads), dtype=native.model.vision.pos_emb.dtype),
    )
    try:
        ov_model = ov.convert_model(decoder, example_input=example)
        names = [port.get_any_name() for port in ov_model.inputs]
        if len(names) != 4:
            raise DecoderExportError(f"Unexpected decoder inputs: {names}")
        cache_input = ov_model.input(3)
        cache_input.get_tensor().set_names({"past_key_values"})
        # Dynamic sequence/cache axes are needed for visual prefill and each
        # following one-token decode invocation.
        ov_model.reshape({
            ov_model.input(0): ov.PartialShape([1, -1, config.dim]),
            ov_model.input(1): ov.PartialShape([-1]),
            ov_model.input(2): ov.PartialShape([1, 1, -1, -1]),
            cache_input: ov.PartialShape([config.n_layers, 2, 1, config.n_kv_heads, ov.Dimension(0, max_context), config.dim // config.n_heads]),
        })
        ov_model.output(0).get_tensor().set_names({"logits"})
        ov_model.output(1).get_tensor().set_names({"present_key_values"})
        _make_stateful(ov_model)
        path = output_dir / "decoder.xml"
        ov.save_model(ov_model, path, compress_to_fp16=(precision == "fp16"))
    except DecoderExportError:
        raise
    except Exception as exc:
        raise DecoderExportError(f"Decoder conversion failed: {exc}") from exc
    metadata = {
        "layers": config.n_layers,
        "heads": config.n_heads,
        "kv_heads": config.n_kv_heads,
        "head_dim": config.dim // config.n_heads,
        "hidden_size": config.dim,
        "vocab_size": config.vocab_size,
        "max_context": max_context,
        "visual_prefix_tokens": 730,
        "state_variable": "past_key_values",
    }
    (output_dir / "decoder_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path
