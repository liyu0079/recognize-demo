"""Optional local enrichment adapters for the six-capability pipeline.

The adapters are deliberately fail-closed: a missing local model, runtime, or
IR produces an empty field and a health warning rather than fabricated OCR or
caption text. Model acquisition/export is kept separate in
``export_all_full_models.py``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from model_downloader import audit_models, startup_audit

logger = logging.getLogger("vision-annotator")


class FullCapabilityEnricher:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.ocr_model_dir = Path(os.getenv("PADDLEOCR_MODEL_DIR", str(model_dir / "paddleocr"))).resolve()
        self.ocr_det_dir = Path(os.getenv("PADDLEOCR_DET_MODEL_DIR", str(self.ocr_model_dir / "det"))).resolve()
        self.ocr_rec_dir = Path(os.getenv("PADDLEOCR_REC_MODEL_DIR", str(self.ocr_model_dir / "rec"))).resolve()
        configured_caption_dir = os.getenv("MOONDREAM_OPENVINO_DIR", "").strip()
        self.caption_model_dir = Path(configured_caption_dir).resolve() if configured_caption_dir else (model_dir / "openvino").resolve()
        self._ocr: Any | None = None
        self._ocr_error: str | None = None
        self._caption: Any | None = None
        self._caption_error: str | None = None
        # 启动时只审计本地文件；AUTO_DOWNLOAD_MODELS=1 时才会下载。
        self.asset_report = startup_audit()

    def _caption_xml(self) -> Path | None:
        for candidate in (self.caption_model_dir / "moondream2.xml", self.caption_model_dir / "openvino.xml", self.caption_model_dir / "moondream.xml"):
            if candidate.exists() and candidate.with_suffix(".bin").exists():
                return candidate
        return None

    def status(self) -> dict[str, Any]:
        # 健康探针刷新本地审计，但不会触发下载或联网。
        self.asset_report = audit_models(auto_download=False)
        return {
            "assets": self.asset_report,
            "ocr": {
                "available": self._ocr_available(),
                "model_dir": str(self.ocr_model_dir),
                "det_model_dir": str(self.ocr_det_dir),
                "rec_model_dir": str(self.ocr_rec_dir),
                "error": self._ocr_error,
            },
            # 健康检查只检查资产与运行时，不触发模型编译；首次框选推理时
            # 才会由 caption() 懒加载 OpenVINO GenAI 管线。
            "caption": {"available": self._caption_xml() is not None, "model_dir": str(self.caption_model_dir), "error": self._caption_error or ("需要 openvino-genai 与 Moondream2 IR" if self._caption_xml() is None else None)},
        }

    def _ocr_available(self) -> bool:
        # 下载的 .tar 不是可推理模型；只有两个归档均已安全解压时才报告可用。
        return (
            self.ocr_det_dir.exists()
            and self.ocr_rec_dir.exists()
            and any(self.ocr_det_dir.glob("*"))
            and any(self.ocr_rec_dir.glob("*"))
        )

    def _load_ocr(self) -> Any | None:
        if self._ocr is not None:
            return self._ocr
        if not self._ocr_available():
            return None
        try:
            from paddleocr import PaddleOCR  # type: ignore
            self._ocr = PaddleOCR(
                lang="ch",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_dir=str(self.ocr_det_dir),
                text_recognition_model_dir=str(self.ocr_rec_dir),
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            self._ocr_error = str(exc)
            logger.warning("PaddleOCR 本地适配器未加载：%s", exc)
        return self._ocr

    def ocr(self, crop: np.ndarray) -> str:
        engine = self._load_ocr()
        if engine is None:
            return ""
        try:
            result = engine.predict(crop) if hasattr(engine, "predict") else engine.ocr(crop, cls=False)
            texts: list[str] = []
            for page in result or []:
                if isinstance(page, dict):
                    texts.extend(str(item) for item in page.get("rec_texts", []) if item)
                elif isinstance(page, list):
                    texts.extend(str(item[1][0]) for item in page if isinstance(item, list) and len(item) > 1 and item[1])
            return " ".join(texts).strip()
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("PaddleOCR 局部识别失败：%s", exc)
            return ""

    def caption(self, crop: np.ndarray) -> str:
        pipe = self._load_caption()
        if pipe is None:
            return ""
        try:
            # OpenVINO GenAI VLM releases expose slightly different argument
            # orderings. Try the two local-only signatures without falling back
            # to a template or remote service.
            prompt = "Describe this image briefly."
            for args in ((prompt, crop), (crop, prompt)):
                try:
                    result = pipe.generate(*args, max_new_tokens=48)
                    text = str(result).strip()
                    if text:
                        return text
                except TypeError:
                    continue
            return ""
        except Exception as exc:  # pragma: no cover - optional dependency
            self._caption_error = str(exc)
            logger.warning("Moondream 局部描述失败：%s", exc)
            return ""

    def _load_caption(self) -> Any | None:
        if self._caption is not None:
            return self._caption
        xml = self._caption_xml()
        if xml is None:
            return None
        try:
            import openvino_genai as ov_genai  # type: ignore
            device = os.getenv("OPENVINO_DEVICE", "GPU.0")
            # VLM pipeline accepts the directory containing the converted
            # Moondream language/vision files; the exact tokenizer layout is
            # supplied by the model exporter and remains entirely local.
            self._caption = ov_genai.VLMPipeline(str(xml.parent), device)
        except Exception as exc:  # pragma: no cover - optional dependency
            self._caption_error = str(exc)
            logger.warning("Moondream OpenVINO GenAI 未加载：%s", exc)
        return self._caption

    def enrich(self, image: np.ndarray, objects: list[Any]) -> list[Any]:
        height, width = image.shape[:2]
        for item in objects:
            x1, y1, x2, y2 = [int(max(0, value)) for value in item.bbox]
            x2, y2 = min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))
            crop = image[y1:y2, x1:x2]
            if not getattr(item, "description", ""):
                coverage = max(0.0, (x2 - x1) * (y2 - y1) / max(1, width * height) * 100)
                item.description = f"检测到 {item.label}，约占画面 {coverage:.1f}%"
            if crop.size:
                item.ocr_text = self.ocr(crop)
                item.caption = self.caption(crop)
        return objects
