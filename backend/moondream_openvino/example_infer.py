"""Run local Moondream2 OpenVINO Caption or visual question answering."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

if __package__:
    from .kv_cache_adapter import StatefulMoondreamPipeline
else:
    from kv_cache_adapter import StatefulMoondreamPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Moondream2 Stateful OpenVINO inference")
    parser.add_argument("--model", required=True, help="directory produced by export_moondream2.py")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="", help="empty means Caption; otherwise visual question answering")
    parser.add_argument("--device", default="GPU.0")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--compare-native", action="store_true", help="also run the local PyTorch snapshot for a deterministic Caption comparison")
    parser.add_argument("--source-model", default="", help="local source snapshot; required with --compare-native")
    args = parser.parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    pipeline = StatefulMoondreamPipeline(args.model, args.device)
    if args.prompt:
        result = pipeline.answer(image, args.prompt, args.max_new_tokens)
    else:
        result = pipeline.generate_caption(image, "short", args.max_new_tokens)
    print(f"device={pipeline.device}")
    print(f"openvino: {result}")
    if args.compare_native:
        if not args.source_model:
            parser.error("--source-model is required with --compare-native")
        from PIL import Image
        if __package__:
            from .runtime import load_native_model
        else:
            from runtime import load_native_model
        native = load_native_model(Path(args.source_model))
        # Native Moondream's Caption interface is deterministic by default.
        native_result = native.caption(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), length="short")
        native_text = str(native_result.get("caption", "")).strip()
        print(f"pytorch: {native_text}")


if __name__ == "__main__":
    main()
