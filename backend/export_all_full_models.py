"""Prepare and audit the complete local six-capability model set.

GroundingDINO, SAM2 and RTMPose-body reuse the project's established exporter.
The downloaded hand model is LiteRT/TFLite and Moondream2 is a local
``safetensors`` snapshot, so they need dedicated conversion attempts rather
than an environment variable pointing at a fictional ONNX file.  If their
upstream formats cannot be converted, the runtime uses a truthful local
LiteRT/PyTorch fallback and this script reports that OpenVINO IR is still
missing.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import openvino as ov

from export_all_models import main as export_core
from model_downloader import audit_models

BASE_DIR = Path(__file__).resolve().parent
IR_DIR = BASE_DIR / "models" / "openvino"
WEIGHTS_DIR = BASE_DIR.parent / "weights"


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


def convert_hand_litert(force: bool = False) -> bool:
    """Attempt to convert the exact downloaded LiteRT hand-pose graph.

    This is deliberately an attempted conversion, not a renamed TFLite file.
    OpenVINO 2025 currently rejects this particular graph on some Windows
    builds.  The caller receives ``False`` and the service switches to the
    real LiteRT local runtime in that case.
    """
    target = IR_DIR / "rtmpose_hand.xml"
    source = WEIGHTS_DIR / "rtmpose-hand-litert" / "rtmhand_fp16.tflite"
    if target.exists() and target.with_suffix(".bin").exists() and not force:
        print(f"[skip] rtmpose_hand: {target}")
        return True
    if not source.is_file():
        print(f"[missing] rtmpose_hand source: {source}")
        return False
    IR_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model = ov.convert_model(str(source))
        ov.save_model(model, target, compress_to_fp16=True)
        print(f"[done] rtmpose_hand: {target}")
        return True
    except Exception as exc:
        print("[fallback] rtmpose_hand: OpenVINO cannot convert the downloaded "
              f"LiteRT graph ({exc}). The service will use ai-edge-litert "
              "for real local 21-point inference; no fake IR was created.")
        return False


def export_moondream2(force: bool = False) -> bool:
    """Export Moondream2 through this repository's dedicated exporter only."""
    source = WEIGHTS_DIR / "moondream2"
    target = IR_DIR / "moondream2"
    decoder_xml = target / "decoder.xml"
    if decoder_xml.is_file() and decoder_xml.with_suffix(".bin").is_file() and not force:
        print(f"[skip] moondream2: {target}")
        return True
    if not (source / "model.safetensors").is_file():
        print(f"[missing] moondream2 source: {source / 'model.safetensors'}")
        return False
    if target.exists() and force:
        # Never delete a valid IR directory before the exporter has succeeded.
        backup = target.with_name("moondream2.previous")
        if backup.exists():
            shutil.rmtree(backup)
        target.replace(backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from moondream_openvino.export_moondream2 import export_bundle
        print(f"[export] moondream2 dedicated Stateful exporter: {source} -> {target}", flush=True)
        export_bundle(source, target, precision="fp16", max_context=2048)
    except Exception as exc:
        print(f"[fallback] moondream2 dedicated export failed: {exc}")
        return False
    if not decoder_xml.is_file() or not decoder_xml.with_suffix(".bin").is_file():
        print(f"[fallback] moondream2: dedicated exporter did not create {decoder_xml}; refusing to mark IR as ready.")
        return False
    print(f"[done] moondream2: {target}")
    return True


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
    def ir_ready(name: str) -> bool:
        if name == "moondream2":
            path = IR_DIR / "moondream2" / "decoder.xml"
        else:
            path = IR_DIR / f"{name}.xml"
        return path.is_file() and path.with_suffix(".bin").is_file()

    missing = [name for name in required if not ir_ready(name)]
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
    parser.add_argument("--moondream-only", action="store_true", help="only export the local Moondream2 snapshot")
    args = parser.parse_args()
    required = ("grounding_dino", "sam2_encoder", "sam2_prompt", "sam2_decoder", "rtmpose_tiny", "rtmpose_hand", "paddleocr", "moondream2")
    if args.check_only:
        raise SystemExit(check(required))
    if args.moondream_only:
        raise SystemExit(0 if export_moondream2(args.force) else 1)
    export_core()
    convert_hand_litert(args.force)
    export_moondream2(args.force)
    paddle_source = os.getenv("PADDLEOCR_ONNX", "")
    if paddle_source:
        convert_onnx(Path(paddle_source), "paddleocr", args.force)
    else:
        print("[info] paddleocr: uses the installed local PaddleOCR runtime; set PADDLEOCR_ONNX only when an OpenVINO OCR export is required.")
    raise SystemExit(check(required))


if __name__ == "__main__":
    main()
