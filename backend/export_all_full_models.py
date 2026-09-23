"""Prepare the complete six-capability OpenVINO model set.

The existing three exporters are reused. Optional hand/OCR/Caption exporters
consume explicitly supplied local ONNX/IR sources; this script never invents a
model file and never downloads a runtime asset implicitly. Use ``--check-only``
on an offline target to audit the deployment bundle.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import openvino as ov

from export_all_models import main as export_core
from model_downloader import audit_models

BASE_DIR = Path(__file__).resolve().parent
IR_DIR = BASE_DIR / "models" / "openvino"


def convert_onnx(source: Path, name: str, force: bool = False) -> None:
    target = IR_DIR / f"{name}.xml"
    if target.exists() and target.with_suffix(".bin").exists() and not force:
        print(f"[skip] {name}: {target}")
        return
    if not source.exists():
        raise FileNotFoundError(f"缺少 {name} ONNX：{source}。请在准备机提供真实社区预训练权重后再导出。")
    IR_DIR.mkdir(parents=True, exist_ok=True)
    model = ov.convert_model(str(source))
    ov.save_model(model, target, compress_to_fp16=True)
    print(f"[done] {name}: {target}")


def check(required: tuple[str, ...]) -> int:
    assets = audit_models(auto_download=False)
    asset_failed = False
    for name, item in assets.items():
        state = item.get("checksum", "missing")
        if not item.get("available", False):
            asset_failed = True
        print(
            f"[asset] {name}: {state}"
            f" md5={item.get('md5') or '-'}"
            f" expected={item.get('expected_md5') or '-'}"
            f" path={item.get('path', '')}",
        )
    missing = [name for name in required if not (IR_DIR / f"{name}.xml").exists() or not (IR_DIR / f"{name}.bin").exists()]
    if missing:
        print("[missing] " + ", ".join(missing))
    if asset_failed:
        print("[missing] one or more downloaded model assets are missing or have failed MD5 validation")
    if missing or asset_failed:
        return 1
    print("[ok] full six-capability IR bundle is present")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and audit the full local OpenVINO vision bundle")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    required = ("grounding_dino", "sam2_encoder", "sam2_prompt", "sam2_decoder", "rtmpose_tiny", "rtmpose_hand", "paddleocr", "moondream2")
    if args.check_only:
        raise SystemExit(check(required))
    export_core()
    sources = {
        "rtmpose_hand": os.getenv("RTMPOSE_HAND_ONNX", ""),
        "paddleocr": os.getenv("PADDLEOCR_ONNX", ""),
        "moondream2": os.getenv("MOONDREAM2_ONNX", ""),
    }
    for name, source in sources.items():
        if source:
            convert_onnx(Path(source), name, args.force)
        else:
            print(f"[missing] {name}: set the corresponding local ONNX environment variable")
    raise SystemExit(check(required))


if __name__ == "__main__":
    main()
