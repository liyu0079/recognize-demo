"""CLI entry point for the dedicated Moondream2 OpenVINO exporter.

No Optimum or generic causal-language-model exporter is used here.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

if __package__:
    from .decoder_exporter import DecoderExportError, export_decoder
    from .runtime import MoondreamLoadError, load_native_model, tokenizer_path
    from .vision_exporter import VisionExportError, export_vision
else:  # Supports `python export_moondream2.py` from this folder on Windows.
    from decoder_exporter import DecoderExportError, export_decoder
    from runtime import MoondreamLoadError, load_native_model, tokenizer_path
    from vision_exporter import VisionExportError, export_vision


def _export_embedding(native: object, output_dir: Path) -> Path:
    import openvino as ov
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TokenEmbedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = native.model.text.wte  # type: ignore[attr-defined]

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return F.embedding(input_ids, self.weight)

    wrapper = TokenEmbedding().eval()
    model = ov.convert_model(wrapper, example_input=torch.zeros((1, 1), dtype=torch.long))
    model.reshape({model.input(0): ov.PartialShape([1, -1])})
    path = output_dir / "token_embedding.xml"
    ov.save_model(model, path, compress_to_fp16=True)
    return path


def export_bundle(
    model_dir: Path,
    output_dir: Path,
    precision: str,
    max_context: int,
) -> dict[str, str]:
    native = load_native_model(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        vision = export_vision(native, output_dir)
        embedding = _export_embedding(native, output_dir)
        decoder = export_decoder(native, output_dir, max_context=max_context, precision=precision)
    except (VisionExportError, DecoderExportError) as exc:
        raise RuntimeError(str(exc)) from exc
    shutil.copy2(tokenizer_path(model_dir), output_dir / "tokenizer.json")
    manifest = {
        "format": "moondream2-openvino-stateful-v1",
        "precision": precision,
        "source": str(model_dir.resolve()),
        "files": {
            "vision_encoder": str(vision["encoder"].name),
            "vision_projector": str(vision["projector"].name),
            "token_embedding": str(embedding.name),
            "decoder": str(decoder.name),
            "tokenizer": "tokenizer.json",
        },
        "runtime": "StatefulMoondreamPipeline",
        "device_preference": "GPU.0",
    }
    (output_dir / "moondream2_ov_config.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return manifest["files"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export local Moondream2 to dedicated Stateful OpenVINO IR")
    parser.add_argument("--model", required=True, help="local vikhyatk/moondream2 snapshot directory")
    parser.add_argument("--output", required=True, help="output directory for XML/BIN bundle")
    parser.add_argument("--precision", choices=("fp16", "int8"), default="fp16")
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--device", default="arc", choices=("arc", "cpu"), help="recorded preference; export itself is device independent")
    parser.add_argument("--quantize", action="store_true", help="requires --precision int8 and a future calibrated quantization extension")
    args = parser.parse_args()
    if args.quantize != (args.precision == "int8"):
        parser.error("--quantize must be used exactly with --precision int8")
    model_dir, output_dir = Path(args.model), Path(args.output)
    if not model_dir.is_dir():
        parser.error("--model must be a local, complete vikhyatk/moondream2 snapshot; network downloads are not performed by the exporter")
    try:
        files = export_bundle(model_dir, output_dir, args.precision, args.max_context)
    except (MoondreamLoadError, RuntimeError) as exc:
        raise SystemExit(f"Moondream2 export failed: {exc}") from exc
    print("Moondream2 Stateful OpenVINO bundle created:")
    for name, value in files.items():
        print(f"  {name}: {output_dir / value}")


if __name__ == "__main__":
    main()
