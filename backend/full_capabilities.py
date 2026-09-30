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

import cv2
import numpy as np
from model_downloader import startup_audit
from local_model_runtimes import LiteRTHandPoseRuntime, LocalMoondreamRuntime

logger = logging.getLogger("vision-annotator")
MAX_CAPTION_OBJECTS = max(0, int(os.getenv("MAX_CAPTION_OBJECTS", "8")))
PADDLEOCR_MIN_SCORE = max(0.0, min(1.0, float(os.getenv("PADDLEOCR_MIN_SCORE", "0.45"))))
OBJECT_NAMES = {
    "person": "人员", "child": "儿童", "man": "男性", "woman": "女性",
    "sign": "招牌", "sign text": "招牌文字", "text": "文字", "license plate": "车牌",
    "bottle": "瓶子", "chair": "椅子", "handbag": "手提包", "motorcycle": "摩托车",
    "car": "车辆", "truck": "卡车", "pickup truck": "皮卡", "SUV": "越野车",
    "bus": "公交车", "tree": "树木", "plant": "植物", "tomato": "西红柿",
    "tomato plant": "西红柿植株", "tomato vine": "西红柿植株", "shoe": "鞋子", "sneaker": "运动鞋", "building": "建筑",
    "manhole cover": "井盖", "air conditioner": "空调外机", "outdoor air conditioner": "空调外机",
    "AC outdoor unit": "空调外机", "HVAC unit": "空调外机", "road": "道路", "dirt road": "土路",
    "guardrail": "护栏", "traffic sign": "交通标志", "street lamp": "路灯", "tire": "轮胎",
    "wheel": "车轮", "desert": "荒野", "field": "旷野", "sand": "沙地", "grass": "草地",
}


class FullCapabilityEnricher:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir.resolve()
        # Existing OCR lives in backend/models. ModelScope's larger raw hand
        # and Moondream snapshots are kept in the project-level weights dir.
        default_weights_dir = self.model_dir.parent.parent / "weights"
        self.weights_dir = Path(os.getenv("MODEL_WEIGHTS_DIR", str(default_weights_dir))).resolve()
        self.ocr_model_dir = Path(os.getenv("PADDLEOCR_MODEL_DIR", str(self.model_dir / "paddleocr"))).resolve()
        self.ocr_det_dir = Path(os.getenv("PADDLEOCR_DET_MODEL_DIR", str(self.ocr_model_dir / "det"))).resolve()
        self.ocr_rec_dir = Path(os.getenv("PADDLEOCR_REC_MODEL_DIR", str(self.ocr_model_dir / "rec"))).resolve()
        configured_caption_dir = os.getenv("MOONDREAM_OPENVINO_DIR", "").strip()
        self.caption_model_dir = Path(configured_caption_dir).resolve() if configured_caption_dir else (self.model_dir / "openvino" / "moondream2").resolve()
        self.moondream_model_dir = Path(os.getenv("MOONDREAM_MODEL_DIR", str(self.weights_dir / "moondream2"))).resolve()
        self.hand_model_path = Path(os.getenv("RTMPOSE_HAND_TFLITE", str(self.weights_dir / "rtmpose-hand-litert" / "rtmhand_fp16.tflite"))).resolve()
        self._ocr: Any | None = None
        self._ocr_error: str | None = None
        self._ocr_load_attempted = False
        self._caption: Any | None = None
        self._caption_error: str | None = None
        self._caption_load_attempted = False
        self._hand = LiteRTHandPoseRuntime(self.hand_model_path)
        self._local_caption = LocalMoondreamRuntime(self.moondream_model_dir)
        # 启动时只审计本地文件；AUTO_DOWNLOAD_MODELS=1 时才会下载。
        self.asset_report = startup_audit()

    def _caption_ir_dir(self) -> Path | None:
        """Return an OpenVINO GenAI export directory rather than one fixed XML.

        VLM exporters produce a set of language/vision XML files, not a single
        conventional ``moondream2.xml``.  Checking only that filename made a
        successful multi-file export look absent.
        """
        candidates = (self.caption_model_dir, self.model_dir / "openvino")
        for directory in candidates:
            if not directory.is_dir():
                continue
            # The dedicated exporter produces decoder.xml plus a manifest,
            # rather than OpenVINO GenAI's registered language-model name.
            if (directory / "moondream2_ov_config.json").is_file() and (directory / "decoder.xml").is_file() and (directory / "decoder.bin").is_file():
                return directory
            xml_files = list(directory.glob("*.xml"))
            has_language = any("language" in item.name or item.name in {"moondream2.xml", "openvino.xml", "moondream.xml"} for item in xml_files)
            if has_language and any(item.with_suffix(".bin").is_file() for item in xml_files):
                return directory
        return None

    def status(self) -> dict[str, Any]:
        # 健康接口由前端每 5 秒轮询。不能在这里重新计算数 GB 权重目录的
        # MD5，否则多个并发轮询会阻塞 Uvicorn 线程并把正常后端误报为离线。
        # 启动阶段和 model_downloader 手动命令仍执行完整审计；此处只返回
        # 最近一次真实审计快照。
        return {
            "assets": self.asset_report,
            "ocr": {
                "available": self._ocr is not None or (not self._ocr_load_attempted and self._ocr_available()),
                "model_dir": str(self.ocr_model_dir),
                "det_model_dir": str(self.ocr_det_dir),
                "rec_model_dir": str(self.ocr_rec_dir),
                "error": self._ocr_error,
            },
            "hand_pose": self._hand.status(),
            # 健康检查只检查资产与运行时，不触发数 GB 的 Caption 懒加载。
            # IR 不存在时，caption() 会使用同一份本地 safetensors 的 CPU
            # 实现，状态中明确标出实际引擎，绝不把 CPU 说成 OpenVINO。
            "caption": {
                **self._local_caption.status(),
                "available": self._caption_ir_dir() is not None or self._local_caption.available,
                "engine": "openvino" if self._caption_ir_dir() is not None else "pytorch-cpu",
                "openvino_ir": self._caption_ir_dir() is not None,
                "openvino_dir": str(self.caption_model_dir),
                "error": self._caption_error or self._local_caption.status()["error"],
            },
        }

    def _ocr_available(self) -> bool:
        # PaddleOCR 3.x/PaddleX 本地模型目录必须带 inference.yml；仅有旧版
        # .pdmodel/.pdiparams 时无法被新运行时加载，不能误报 OCR 已可用。
        def complete(directory: Path) -> bool:
            return (
                (directory / "inference.yml").is_file()
                and (directory / "inference.pdiparams").is_file()
                and ((directory / "inference.pdmodel").is_file() or (directory / "inference.json").is_file())
            )

        return complete(self.ocr_det_dir) and complete(self.ocr_rec_dir)

    def _load_ocr(self) -> Any | None:
        if self._ocr is not None:
            return self._ocr
        # PaddleOCR 版本/本地模型目录不兼容时，避免每个检测框都重复初始化，
        # 造成数十秒卡顿和同一条警告刷屏。
        if self._ocr_load_attempted:
            return None
        self._ocr_load_attempted = True
        if not self._ocr_available():
            self._ocr_error = "本地 PaddleOCR det/rec 模型不完整：每个目录需要 inference.yml、inference.pdiparams 与推理模型文件"
            return None
        try:
            from paddleocr import PaddleOCR  # type: ignore
            self._ocr = PaddleOCR(
                # 本地目录由 PaddleOCR 3.x 官方 PP-OCRv4 mobile 模型提供。
                # 不明确模型名时新版会默认选择 v6，造成配置和本地权重不匹配。
                text_detection_model_name="PP-OCRv4_mobile_det",
                text_recognition_model_name="PP-OCRv4_mobile_rec",
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
            return " ".join(text for text, _ in self._ocr_regions(crop)).strip()
        except Exception as exc:  # pragma: no cover - optional dependency
            self._ocr_load_attempted = True
            self._ocr = None
            self._ocr_error = str(exc)
            logger.warning("PaddleOCR 局部识别失败：%s", exc)
            return ""

    def _ocr_regions(self, image: np.ndarray) -> list[tuple[str, tuple[float, float, float, float]]]:
        """单次整图 OCR，返回文字和其像素框，避免逐检测框重复前向。"""
        engine = self._load_ocr()
        if engine is None:
            return []
        result = engine.predict(image) if hasattr(engine, "predict") else engine.ocr(image, cls=False)
        regions: list[tuple[str, tuple[float, float, float, float]]] = []
        for page in result or []:
            page_data: Any = page.json if hasattr(page, "json") else page
            if isinstance(page_data, dict) and isinstance(page_data.get("res"), dict):
                page_data = page_data["res"]
            if isinstance(page_data, dict):
                texts = page_data.get("rec_texts", [])
                boxes = page_data.get("rec_boxes", page_data.get("dt_polys", []))
                scores = page_data.get("rec_scores", [])
                for index, (text, box) in enumerate(zip(texts, boxes)):
                    try:
                        score = float(scores[index]) if index < len(scores) else None
                        # OCR 返回了置信度时过滤明显噪声；旧 Paddle 结构没有分数时不臆造分数。
                        if score is not None and score < PADDLEOCR_MIN_SCORE:
                            continue
                        coords = np.asarray(box, dtype=np.float32).reshape(-1, 2)
                        x1, y1 = coords.min(axis=0)
                        x2, y2 = coords.max(axis=0)
                        value = str(text).strip()
                        if value and x2 > x1 and y2 > y1:
                            regions.append((value, (float(x1), float(y1), float(x2), float(y2))))
                    except (TypeError, ValueError):
                        continue
            elif isinstance(page, list):
                for item in page:
                    if isinstance(item, list) and len(item) > 1 and item[1]:
                        value = str(item[1][0]).strip()
                        try:
                            score = float(item[1][1]) if len(item[1]) > 1 else None
                        except (TypeError, ValueError):
                            score = None
                        if score is not None and score < PADDLEOCR_MIN_SCORE:
                            continue
                        coords = np.asarray(item[0], dtype=np.float32).reshape(-1, 2)
                        x1, y1 = coords.min(axis=0)
                        x2, y2 = coords.max(axis=0)
                        if value:
                            regions.append((value, (float(x1), float(y1), float(x2), float(y2))))
        return regions

    @staticmethod
    def _ocr_region_label(
        bbox: tuple[float, float, float, float], objects: list[Any],
    ) -> tuple[str, str]:
        """结合局部框与父物体位置，把可信的车牌/招牌文字从泛 text 中区分出来。"""
        x1, y1, x2, y2 = bbox
        text_area = max(1.0, (x2 - x1) * (y2 - y1))
        candidates: list[tuple[float, float, str, tuple[float, float, float, float]]] = []
        for item in objects:
            try:
                px1, py1, px2, py2 = (float(value) for value in item.bbox)
            except (TypeError, ValueError):
                continue
            intersection = max(0.0, min(px2, x2) - max(px1, x1)) * max(0.0, min(py2, y2) - max(py1, y1))
            contained_ratio = intersection / text_area
            parent_area = max(1.0, (px2 - px1) * (py2 - py1))
            if contained_ratio >= 0.75 and parent_area > text_area * 1.2:
                candidates.append((parent_area, -contained_ratio, str(item.label).lower().strip(), (px1, py1, px2, py2)))
        if not candidates:
            return "text", "本地 OCR 识别出的文字区域"

        # 选择最小的可靠包含框，避免楼宇大框抢走招牌/车辆等语义上下文。
        _, _, parent_label, parent = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
        if any(word in parent_label for word in ("sign", "billboard", "storefront sign", "招牌", "标志")):
            return "sign text", "招牌上的文字"

        vehicle_labels = {"car", "truck", "pickup truck", "suv", "van", "bus", "motorcycle"}
        if parent_label in vehicle_labels:
            px1, py1, px2, py2 = parent
            parent_width, parent_height = max(1.0, px2 - px1), max(1.0, py2 - py1)
            region_width, region_height = max(1.0, x2 - x1), max(1.0, y2 - y1)
            center_y = (y1 + y2) / 2
            relative_y = (center_y - py1) / parent_height
            width_ratio = region_width / parent_width
            height_ratio = region_height / parent_height
            # 车牌通常是车辆下半部的横向小区域；不把车窗上的长段 OCR 误标成车牌。
            if region_width / region_height >= 1.35 and 0.12 <= width_ratio <= 0.82 and height_ratio <= 0.35 and relative_y >= 0.48:
                return "license plate", "车牌文字"
        return "text", "本地 OCR 识别出的文字区域"

    @staticmethod
    def _text_for_object(regions: list[tuple[str, tuple[float, float, float, float]]], bbox: tuple[int, int, int, int]) -> str:
        ox1, oy1, ox2, oy2 = bbox
        values: list[str] = []
        for text, (tx1, ty1, tx2, ty2) in regions:
            intersection = max(0.0, min(ox2, tx2) - max(ox1, tx1)) * max(0.0, min(oy2, ty2) - max(oy1, ty1))
            text_area = max(1.0, (tx2 - tx1) * (ty2 - ty1))
            center_inside = ox1 <= (tx1 + tx2) / 2 <= ox2 and oy1 <= (ty1 + ty2) / 2 <= oy2
            if center_inside or intersection / text_area >= 0.25:
                values.append(text)
        return " ".join(dict.fromkeys(values))

    @staticmethod
    def _measured_description(label: str, crop: np.ndarray) -> str:
        """使用检测类别和 ROI 主色给出简短描述，不把框面积误当作语义信息。"""
        key = label.strip()
        name = OBJECT_NAMES.get(key, OBJECT_NAMES.get(key.lower(), key.replace("_", " ")))
        color_name = ""
        if crop.size:
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            # 人物只看躯干中段，避免肤色/天空把衣物颜色误算进去。
            if label.lower() in {"person", "child", "man", "woman"}:
                height = hsv.shape[0]
                hsv = hsv[int(height * 0.25):max(int(height * 0.25) + 1, int(height * 0.78))]
            colorful = (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 55)
            ranges = (
                ("红色", ((0, 10), (170, 179))), ("橙色", ((11, 24),)),
                ("黄色", ((25, 38),)), ("绿色", ((39, 88),)),
                ("蓝色", ((89, 132),)), ("紫色", ((133, 169),)),
            )
            best_name, best_ratio = "", 0.0
            for candidate, hue_ranges in ranges:
                matched = np.zeros_like(colorful, dtype=bool)
                for low, high in hue_ranges:
                    matched |= (hsv[:, :, 0] >= low) & (hsv[:, :, 0] <= high)
                ratio = float((matched & colorful).mean())
                if ratio > best_ratio:
                    best_name, best_ratio = candidate, ratio
            if best_ratio >= 0.28:
                color_name = best_name
            if not color_name:
                white = (hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 170)
                gray = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] >= 75) & (hsv[:, :, 2] <= 190)
                if float(white.mean()) >= 0.48:
                    color_name = "白色"
                elif float(gray.mean()) >= 0.5:
                    color_name = "灰色"
        if key.lower() in {"person", "child", "man", "woman"}:
            return f"穿{color_name}衣物的{name}" if color_name else name
        return f"{color_name}{name}" if color_name else name

    def caption(self, crop: np.ndarray) -> str:
        pipe = self._load_caption()
        if pipe is not None:
            try:
                if hasattr(pipe, "generate_caption"):
                    # Enricher crops are RGB; the dedicated vision IR accepts
                    # BGR uint8 to include BGR-to-RGB conversion in its graph.
                    return str(pipe.generate_caption(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))).strip()
                # OpenVINO GenAI VLM releases expose slightly different
                # argument orderings. Try the two local-only signatures.
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
                logger.warning("Moondream OpenVINO 局部描述失败，将尝试本地 CPU：%s", exc)
        return self._local_caption.caption(crop)

    def _load_caption(self) -> Any | None:
        if self._caption is not None:
            return self._caption
        # 缺失 IR 或运行时不兼容时，不要为每一个检测框重复尝试加载 VLM。
        if self._caption_load_attempted:
            return None
        self._caption_load_attempted = True
        ir_dir = self._caption_ir_dir()
        if ir_dir is None:
            return None
        try:
            device = os.getenv("OPENVINO_DEVICE", "GPU.0")
            if (ir_dir / "moondream2_ov_config.json").is_file():
                from moondream_openvino.kv_cache_adapter import StatefulMoondreamPipeline
                self._caption = StatefulMoondreamPipeline(ir_dir, device)
            else:
                import openvino_genai as ov_genai  # type: ignore
                # VLM pipeline accepts a registered converted VLM directory.
                self._caption = ov_genai.VLMPipeline(str(ir_dir), device)
        except Exception as exc:  # pragma: no cover - optional dependency
            self._caption_error = str(exc)
            logger.warning("Moondream OpenVINO GenAI 未加载：%s", exc)
        return self._caption

    def enrich(self, image: np.ndarray, objects: list[Any], object_factory: Any | None = None) -> list[Any]:
        height, width = image.shape[:2]
        # OCR 独立覆盖整张图；即使 GroundingDINO 没有返回物体，也不能因此跳过文字识别。
        try:
            ocr_regions = self._ocr_regions(image)
        except Exception as exc:  # pragma: no cover - optional dependency
            self._ocr_error = str(exc)
            logger.warning("PaddleOCR 整图识别失败：%s", exc)
            ocr_regions = []

        # OCR 的检测框本身是最小可追溯文字区域。除向相交的物体框回填文字
        # 外，还返回独立 text 实例，前端可像 DINO-X 一样在车牌、招牌等文字
        # 上直接绘制最小标注框，而不会让一整辆车或整块招牌承担文字标签。
        text_objects: list[Any] = []
        object_type = object_factory or (type(objects[0]) if objects else None)
        if object_type is not None:
            for text, (x1, y1, x2, y2) in ocr_regions:
                left = max(0.0, min(float(x1), float(width - 1)))
                top = max(0.0, min(float(y1), float(height - 1)))
                right = max(left + 1.0, min(float(x2), float(width)))
                bottom = max(top + 1.0, min(float(y2), float(height)))
                label, description = self._ocr_region_label((left, top, right, bottom), objects)
                try:
                    text_objects.append(object_type(
                        label=label,
                        # PaddleOCR 3.x 的返回结构没有统一的区域置信度字段；
                        # 不伪造检测分数，文本区域使用 0 作为“未提供”哨兵值。
                        score=0.0,
                        bbox=[left, top, right, bottom],
                        mask="",
                        ocr_text=text,
                        description=description,
                        caption="",
                    ))
                except (TypeError, ValueError) as exc:
                    logger.warning("OCR 文字框无法写入统一结果：%s", exc)
        for index, item in enumerate(objects):
            x1, y1, x2, y2 = [int(max(0, value)) for value in item.bbox]
            x2, y2 = min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))
            crop = image[y1:y2, x1:x2]
            item.description = self._measured_description(str(item.label), crop)
            item.ocr_text = self._text_for_object(ocr_regions, (x1, y1, x2, y2))
            if str(item.label).lower() in {"hand", "hands", "手", "手部"} and not getattr(item, "hand_keypoints", []):
                item.hand_keypoints = [self._hand.estimate(image, np.asarray(item.bbox, dtype=np.float32))]
                item.hand_keypoints = [points for points in item.hand_keypoints if len(points) == 21]
            # Caption 真实推理成本高，只覆盖最高优先级对象；其余框继续返回
            # 真实检测/OCR/几何描述，不以模板冒充模型 Caption。
            if crop.size and index < MAX_CAPTION_OBJECTS:
                item.caption = self.caption(crop)
        return objects + text_objects
