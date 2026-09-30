"""Stateful OpenVINO runtime for the custom Moondream2 visual-prefix cache.

The cache itself is owned by the OpenVINO InferRequest via ReadValue/Assign.
This adapter supplies only the mathematically required position IDs and causal
mask; it never copies or concatenates a KV cache in Python.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class MoondreamRuntimeError(RuntimeError):
    pass


def _mask(past_length: int, query_length: int, prefix_length: int) -> np.ndarray:
    """Moondream's bidirectional visual prefix plus causal text mask."""
    query_positions = np.arange(past_length, past_length + query_length)[:, None]
    key_positions = np.arange(past_length + query_length)[None, :]
    visible = key_positions <= query_positions
    visual_to_visual = (query_positions < prefix_length) & (key_positions < prefix_length)
    return np.logical_or(visible, visual_to_visual)[None, None, :, :]


class StatefulMoondreamPipeline:
    """End-to-end local image + prompt generation on OpenVINO GPU/CPU.

    This uses OpenVINO's state API directly. ``openvino_genai.LLMPipeline``
    cannot accept an arbitrary visual embedding prefix for an unregistered
    custom architecture, while this adapter preserves the same Stateful cache
    semantics and reset operation without exposing cache tensors to callers.
    """

    def __init__(self, model_dir: str | Path, device: str = "GPU.0") -> None:
        import openvino as ov
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir).resolve()
        config_path = self.model_dir / "decoder_config.json"
        tokenizer_path = self.model_dir / "tokenizer.json"
        required = ("vision_encoder.xml", "vision_projector.xml", "token_embedding.xml", "decoder.xml")
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        if missing or not config_path.is_file() or not tokenizer_path.is_file():
            raise MoondreamRuntimeError(f"Incomplete Moondream OpenVINO bundle: {missing}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.core = ov.Core()
        try:
            self.vision = self.core.compile_model(str(self.model_dir / "vision_encoder.xml"), device)
            self.projector = self.core.compile_model(str(self.model_dir / "vision_projector.xml"), device)
            self.embedding = self.core.compile_model(str(self.model_dir / "token_embedding.xml"), device)
            self.decoder = self.core.compile_model(str(self.model_dir / "decoder.xml"), device)
            self.device = device
        except Exception:
            # Arc driver/plugin failures must not turn into a fabricated result.
            self.vision = self.core.compile_model(str(self.model_dir / "vision_encoder.xml"), "CPU")
            self.projector = self.core.compile_model(str(self.model_dir / "vision_projector.xml"), "CPU")
            self.embedding = self.core.compile_model(str(self.model_dir / "token_embedding.xml"), "CPU")
            self.decoder = self.core.compile_model(str(self.model_dir / "decoder.xml"), "CPU")
            self.device = "CPU"
        self.request = self.decoder.create_infer_request()
        self.prefix_tokens = int(self.config["visual_prefix_tokens"])
        self.position = 0

    @staticmethod
    def _first(compiled: Any, values: list[np.ndarray]) -> np.ndarray:
        result = compiled(values)
        return np.asarray(next(iter(result.values())))

    def reset(self) -> None:
        """Reset all OpenVINO state variables before a new image/prompt."""
        self.request.reset_state()
        # Some OpenVINO 2024 releases initialize a dynamic ReadValue variable
        # with the trace example's length-one cache. Replace only on reset so
        # prefill starts from a true zero-length visual/text history.
        for state in self.request.query_state():
            state.set_state(np.zeros((
                self.config["layers"], 2, 1, self.config["kv_heads"], 0,
                self.config["head_dim"],
            ), dtype=np.float16))
        self.position = 0

    def _embed(self, token_ids: list[int]) -> np.ndarray:
        values = np.asarray([token_ids], dtype=np.int64)
        return self._first(self.embedding, [values])

    def image_prefix(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run visual IR. Input is BGR uint8 NCHW; IR owns preprocessing."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise MoondreamRuntimeError("image must be a BGR HWC array")
        tensor = np.ascontiguousarray(image_bgr.transpose(2, 0, 1)[None].astype(np.uint8))
        features = self._first(self.vision, [tensor])
        # Native 378 path creates a global crop plus a 1x1 local tile. At this
        # resolution both feature grids are equivalent after reconstruction.
        return self._first(self.projector, [features, features])

    def _infer(self, embeds: np.ndarray) -> np.ndarray:
        length = int(embeds.shape[1])
        positions = np.arange(self.position, self.position + length, dtype=np.int64)
        mask = _mask(self.position, length, self.prefix_tokens)
        self.request.set_tensor(self.decoder.input(0), embeds)
        self.request.set_tensor(self.decoder.input(1), positions)
        self.request.set_tensor(self.decoder.input(2), mask)
        self.request.infer()
        self.position += length
        return np.asarray(self.request.get_output_tensor(0).data)

    def generate_caption(self, image_bgr: np.ndarray, length: str = "short", max_new_tokens: int = 48) -> str:
        templates = {
            "short": [1, 32708, 2, 12492, 3],
            "normal": [1, 32708, 2, 6382, 3],
            "long": [1, 32708, 2, 4059, 3],
        }
        if length not in templates:
            raise MoondreamRuntimeError("length must be short, normal, or long")
        self.reset()
        visual = self.image_prefix(image_bgr)
        bos = self._embed([0])
        logits = self._infer(np.concatenate([bos, visual, self._embed(templates[length])], axis=1))
        generated: list[int] = []
        next_token = int(np.argmax(logits[0]))
        for _ in range(max_new_tokens):
            if next_token == 0:
                break
            generated.append(next_token)
            logits = self._infer(self._embed([next_token]))
            next_token = int(np.argmax(logits[0]))
        return self.tokenizer.decode(generated).strip()

    def answer(self, image_bgr: np.ndarray, question: str, max_new_tokens: int = 64) -> str:
        self.reset()
        visual = self.image_prefix(image_bgr)
        query_tokens = [1, 15381, 2] + self.tokenizer.encode(question).ids + [3]
        logits = self._infer(np.concatenate([self._embed([0]), visual, self._embed(query_tokens)], axis=1))
        generated: list[int] = []
        next_token = int(np.argmax(logits[0]))
        for _ in range(max_new_tokens):
            if next_token == 0:
                break
            generated.append(next_token)
            logits = self._infer(self._embed([next_token]))
            next_token = int(np.argmax(logits[0]))
        return self.tokenizer.decode(generated).strip()
