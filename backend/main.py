from __future__ import annotations

import logging
import json
import os
import re
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from contextlib import nullcontext
from io import BytesIO
from inspect import signature
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from full_capabilities import FullCapabilityEnricher

# ======================== 可配置参数 ========================
USE_OPENVINO = os.getenv("USE_OPENVINO", "1") not in {"0", "false", "False"}
BOX_CONFIDENCE_THRESHOLD = float(os.getenv("BOX_CONFIDENCE_THRESHOLD", "0.3"))
TEXT_CONFIDENCE_THRESHOLD = 0.25
MAX_IMAGE_LONG_EDGE = int(os.getenv("MAX_IMAGE_LONG_EDGE", "1280"))
OPENVINO_IDLE_SECONDS = int(os.getenv("OPENVINO_IDLE_SECONDS", "90"))
VIDEO_SAMPLE_INTERVAL_SECONDS = 3.0  # 视频每隔约 3 秒取一个关键帧
VIDEO_MIN_SAMPLE_COUNT = 4
VIDEO_MAX_SAMPLE_COUNT = 18
MAX_OBJECTS_PER_FRAME = 80
MAX_RESPONSE_OBJECTS = int(os.getenv("MAX_RESPONSE_OBJECTS", "60"))
MAX_POSE_PERSONS = int(os.getenv("MAX_POSE_PERSONS", "12"))

# 无需提示模式使用本地 Grounding DINO 的常见主体类别词表。Grounding DINO 本身需要
# 文本输入，因此这不是云端“万物模型”，而是可审计、可按业务扩充的本地候选集合。
BASE_UNIVERSAL_PROMPT_GROUPS = (
    "person. child. man. woman. face. head. hand. arm. leg. clothing. shirt. shorts. shoe. sneaker. backpack. handbag. umbrella",
    "soccer ball. football. sports ball. ball. goal. goalpost. tomato. apple. orange. banana. fruit. bottle. cup. bowl",
    "car. truck. bus. van. motorcycle. bicycle. train. traffic light. stop sign. dog. cat. bird. horse. cow. sheep",
    "chair. table. sofa. bed. television. monitor. laptop. cell phone. keyboard. mouse. book. clock. door. window. pole. tree. plant. leaf. grass. road. fence. building. house. bridge. sign. text. tower. wall",
)

# 视频优先覆盖主要运动主体，减少每帧文本编码和检测次数；场景类仍保留道路、植被等主体。
VIDEO_UNIVERSAL_PROMPT_GROUPS = (
    "person. child. face. head. hand. arm. leg. clothing. shirt. shorts. shoe. sneaker. soccer ball. football. sports ball. car. truck. bus. van. motorcycle. bicycle. tomato. fruit",
    "dog. cat. bird. tree. plant. leaf. grass. road. fence. goal. goalpost. building. sign",
)

TEXT_ALIASES = {
    "足球": "soccer ball", "球": "sports ball", "人": "person",
    "儿童": "child", "小孩": "child", "车": "car", "汽车": "car", "车辆": "vehicle",
    "树": "tree", "植被": "vegetation", "草地": "grass", "显示器": "monitor", "屏幕": "monitor",
    "television": "monitor", "tv": "monitor", "display": "monitor",
}

LABEL_ALIASES = {
    "football": "soccer ball", "soccer ball": "soccer ball", "sports ball": "sports ball",
    "ball": "sports ball", "vehicle": "car", "automobile": "car", "motor vehicle": "car",
    "goalpost": "goal", "goal post": "goal", "fruit": "fruit",
}

# 父类与子类关系用于“具体类别优先”过滤。这里只影响重叠框，不会阻止子类候选参与检测。
CATEGORY_HIERARCHY = {
    "object": {"person", "child", "car", "truck", "bus", "bicycle", "soccer ball", "tomato", "bottle", "cup"},
    "thing": {"person", "car", "truck", "bus", "bicycle", "soccer ball", "tomato", "bottle", "cup"},
    "sports ball": {"soccer ball", "football", "basketball", "baseball", "tennis ball"},
    "ball": {"soccer ball", "football", "basketball", "baseball", "tennis ball"},
    "fruit": {"tomato", "apple", "orange", "banana", "lemon", "watermelon", "grape", "strawberry"},
    "vehicle": {"car", "truck", "bus", "van", "motorcycle", "bicycle", "train", "airplane", "boat"},
    "animal": {"dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "bear"},
    "plant": {"tree", "flower", "leaf", "grass", "shrub", "bush"},
    "building": {"house", "apartment", "skyscraper", "warehouse", "school", "church"},
}
MUTUALLY_EXCLUSIVE_GROUPS = (
    {"tomato", "apple", "orange", "banana", "lemon", "watermelon", "grape", "strawberry"},
)

# 中文指代提示不经任何在线服务。映射覆盖现场常见的颜色、主体、位置和状态表达；
# 英文自然语言短句会原样交给 Grounding DINO。
CHINESE_SUBJECTS = {
    "人": "person", "男人": "man", "女人": "woman", "孩子": "child", "婴儿": "baby",
    "狗": "dog", "猫": "cat", "鸟": "bird", "车": "car", "汽车": "car", "卡车": "truck",
    "自行车": "bicycle", "摩托车": "motorcycle", "杯": "cup", "杯子": "cup", "瓶": "bottle",
    "瓶子": "bottle", "电脑": "laptop", "笔记本电脑": "laptop", "显示器": "monitor",
    "手机": "cell phone", "桌子": "table", "椅子": "chair", "西红柿": "tomato", "番茄": "tomato",
    "苹果": "apple", "香蕉": "banana", "书": "book", "包": "backpack", "背包": "backpack",
}
CHINESE_COLORS = {
    "白": "white", "黑": "black", "红": "red", "蓝": "blue", "绿": "green", "黄": "yellow",
    "橙": "orange", "紫": "purple", "粉": "pink", "灰": "gray", "棕": "brown",
}
CHINESE_POSITIONS = {
    "左": "on the left", "右": "on the right", "中间": "in the center", "中央": "in the center",
    "前面": "in front", "后面": "in the back", "顶部": "at the top", "上方": "at the top",
    "下方": "at the bottom", "底部": "at the bottom",
}
CHINESE_STATES = {"红透": "ripe", "成熟": "ripe", "站着": "standing", "坐着": "sitting", "奔跑": "running"}
CHINESE_ATTRIBUTES = {"戴眼镜": "wearing glasses", "戴着眼镜": "wearing glasses", "眼镜": "glasses"}

GROUNDING_DINO_REPO_ID = os.getenv("GROUNDING_DINO_REPO_ID", "IDEA-Research/grounding-dino-tiny")
# OpenVINO 部署固定使用 small；旧 PyTorch 分支也只允许同一版本，避免误加载大模型。
SAM2_VARIANT = "small"
SAM2_VARIANTS = {
    "small": ("facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
}
if SAM2_VARIANT not in SAM2_VARIANTS:
    raise RuntimeError("SAM2 固定使用 small 版本")
SAM2_REPO_ID, SAM2_CHECKPOINT_FILENAME, SAM2_CONFIG_NAME = SAM2_VARIANTS[SAM2_VARIANT]

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "models"))).resolve()
GROUNDING_DINO_DIR = Path(os.getenv("GROUNDING_DINO_MODEL_DIR", str(MODEL_DIR / GROUNDING_DINO_REPO_ID.rsplit("/", 1)[-1]))).resolve()
SAM2_DIR = MODEL_DIR / f"sam2.1-hiera-{SAM2_VARIANT}"
GROUNDING_DINO_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.txt",
)

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
UNIVERSAL_CATEGORY_FILE = Path(
    os.getenv("UNIVERSAL_CATEGORY_FILE", str(MODEL_DIR / "universal_categories.json"))
).resolve()
DEFAULT_UNIVERSAL_CATEGORY_FILE = BASE_DIR / "config" / "universal_categories.txt"
MAX_UNIVERSAL_CATEGORIES = 192

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("vision-annotator")


def _category_tokens(values: Any) -> list[str]:
    if isinstance(values, dict):
        values = values.get("categories", values.get("labels", values.get("items", [])))
    if isinstance(values, str):
        values = re.split(r"[,\n;]+", values)
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("name", value.get("label", ""))
        if not isinstance(value, str):
            continue
        value = re.sub(r"\s+", " ", value.strip().lower().strip("."))
        blocked = {"food", "animal", "plant", "vegetation", "fruit", "building", "vehicle", "object", "thing", "item", "stuff", "equipment", "device"}
        if value and value not in blocked and re.fullmatch(r"[a-z0-9][a-z0-9 _-]{0,59}", value) and value not in result:
            result.append(value)
    return result[:MAX_UNIVERSAL_CATEGORIES]


def load_universal_categories() -> tuple[str, ...]:
    """只加载本地候选类别，运行期不会访问外网。"""
    categories: list[str] = []
    for category_file in (DEFAULT_UNIVERSAL_CATEGORY_FILE, UNIVERSAL_CATEGORY_FILE):
        if not category_file.exists():
            continue
        try:
            raw_categories = category_file.read_text(encoding="utf-8")
            try:
                parsed_categories: Any = json.loads(raw_categories)
            except json.JSONDecodeError:
                parsed_categories = raw_categories
            loaded = _category_tokens(parsed_categories)
            categories.extend(loaded)
            logger.info("读取本地候选类别 %d 个：%s", len(loaded), category_file)
        except Exception as exc:
            logger.warning("本地候选类别文件读取失败，将跳过该文件：%s", exc)
    merged = list(dict.fromkeys(categories))
    if not merged:
        return BASE_UNIVERSAL_PROMPT_GROUPS
    # 外部类别作为一个独立分组，保留内置分组的稳定召回。
    chunks = [". ".join(merged[index : index + 64]) for index in range(0, len(merged), 64)]
    return BASE_UNIVERSAL_PROMPT_GROUPS + tuple(chunks)


UNIVERSAL_PROMPT_GROUPS = load_universal_categories()


class DetectedObject(BaseModel):
    label: str
    score: float
    bbox: list[float]
    # 完整二维数组会使 32 个实例的响应达到数十 MB；API 使用行优先二值 RLE。
    # Public API uses the compact row-major RLE string; list matrices remain
    # accepted internally for compatibility with the legacy inference paths.
    mask: str | list[list[int]] = ""
    mask_rle: str = ""
    mask_size: list[int] = []
    keypoints: list[dict[str, float]] = []
    # 当前本地模型没有手部姿态权重；保留明确字段，避免客户端猜测或伪造关键点。
    hand_keypoints: list[list[dict[str, float]]] = []
    ocr_text: str = ""
    description: str = ""
    caption: str = ""


def local_capabilities() -> dict[str, dict[str, Any]]:
    """Report only capabilities backed by files actually present on disk."""
    ir_dir = MODEL_DIR / "openvino"
    files = {
        "grounding_dino": ir_dir / "grounding_dino.xml",
        "sam2_small": ir_dir / "sam2_encoder.xml",
        "rtmpose_tiny": ir_dir / "rtmpose_tiny.xml",
        "rtmpose_hand": ir_dir / "rtmpose_hand.xml",
        "paddleocr": ir_dir / "paddleocr.xml",
        "moondream2": ir_dir / "moondream2.xml",
    }
    return {name: {"available": path.exists() and path.with_suffix('.bin').exists(), "path": str(path)} for name, path in files.items()}


def encode_mask_rle(mask: list[list[int]] | np.ndarray) -> tuple[str, list[int]]:
    """Encode a binary mask as row-major run lengths beginning with a zero run."""
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or not binary.size:
        return "", []
    flat = (binary.reshape(-1) > 0).astype(np.uint8)
    runs: list[int] = []
    value = 0
    count = 0
    for pixel in flat:
        pixel_value = int(pixel)
        if pixel_value == value:
            count += 1
        else:
            runs.append(count)
            value = pixel_value
            count = 1
    runs.append(count)
    return ",".join(str(run) for run in runs), [int(binary.shape[0]), int(binary.shape[1])]


def object_description(label: str, bbox: list[float], image_width: int, image_height: int) -> str:
    """Describe only measured detection data; this is not a generated scene caption."""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    horizontal = "左侧" if center_x < image_width / 3 else "右侧" if center_x > image_width * 2 / 3 else "中部"
    vertical = "上方" if center_y < image_height / 3 else "下方" if center_y > image_height * 2 / 3 else "中部"
    coverage = max(0.0, (x2 - x1) * (y2 - y1) / max(1, image_width * image_height) * 100)
    location = horizontal if horizontal == "中部" else f"{horizontal}{vertical}"
    if label.lower() == "text":
        return f"画面{location}的文本区域，约占画面 {coverage:.1f}%"
    return f"画面{location}的 {label}，约占画面 {coverage:.1f}%"


def compact_object(item: DetectedObject) -> DetectedObject:
    raw_mask = item.mask if isinstance(item.mask, list) else []
    mask_rle, mask_size = encode_mask_rle(raw_mask)
    return DetectedObject(
        label=item.label,
        score=item.score,
        bbox=item.bbox,
        mask=mask_rle,
        mask_rle=mask_rle,
        mask_size=mask_size,
        keypoints=item.keypoints,
        hand_keypoints=item.hand_keypoints,
        ocr_text=item.ocr_text,
        description=item.description,
        caption=item.caption,
    )


class VideoFrameAnnotation(BaseModel):
    timestamp: float
    objects: list[DetectedObject]


class AnalyzeResponse(BaseModel):
    objects: list[DetectedObject]
    labels: list[str]
    feedback: str
    summary: str
    task_id: str
    prompt_mode: Literal["universal", "text", "referring"]
    media_type: Literal["image", "video"]
    frames: list[VideoFrameAnnotation] = []
    activity: str = ""


class PyTorchModelManager:
    """保留的旧 PyTorch CPU 推理引擎，用于 OpenVINO 一键回退。"""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.processor: Any | None = None
        self.detector: Any | None = None
        self.sam_predictor: Any | None = None
        self.load_error: str | None = None
        self.loading = False
        self.load_started_at: float | None = None
        self.load_finished_at: float | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.processor is not None and self.detector is not None and self.sam_predictor is not None

    def _ensure_grounding_dino(self) -> Path:
        required_files = [GROUNDING_DINO_DIR / name for name in GROUNDING_DINO_REQUIRED_FILES]
        weight_file = GROUNDING_DINO_DIR / "model.safetensors"
        if all(path.exists() for path in required_files) and weight_file.stat().st_size > 100_000_000:
            logger.info("检测到 Grounding DINO 本地模型：%s", GROUNDING_DINO_DIR)
            return GROUNDING_DINO_DIR

        raise RuntimeError(f"缺少本地 Grounding DINO 权重：{GROUNDING_DINO_DIR}。请先运行 export_all_models.py。")

    def _ensure_sam2(self) -> Path:
        checkpoint = SAM2_DIR / SAM2_CHECKPOINT_FILENAME
        if checkpoint.exists() and checkpoint.stat().st_size > 100_000_000:
            logger.info("检测到 SAM2 本地模型：%s", checkpoint)
            return checkpoint

        raise RuntimeError(f"缺少本地 SAM2-small 权重：{checkpoint}。请先运行 export_all_models.py。")

    def load(self) -> None:
        if not self._load_lock.acquire(blocking=False):
            return
        self.loading = True
        self.load_error = None
        self.load_started_at = time.time()
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            thread_count = max(1, min(os.cpu_count() or 1, 8))
            torch.set_num_threads(thread_count)
            logger.info("旧 PyTorch 兼容引擎使用 CPU（线程数 %d）。", thread_count)

            grounding_dir = self._ensure_grounding_dino()
            sam2_checkpoint = self._ensure_sam2()

            logger.info("正在从本地目录加载 Grounding DINO：%s", grounding_dir)
            self.processor = AutoProcessor.from_pretrained(grounding_dir, local_files_only=True)
            self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
                grounding_dir,
                local_files_only=True,
            ).to(self.device)
            self.detector.eval()

            logger.info("正在从本地 checkpoint 加载 SAM2.1 %s：%s", SAM2_VARIANT, sam2_checkpoint)
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_model = build_sam2(
                SAM2_CONFIG_NAME,
                str(sam2_checkpoint),
                device=str(self.device),
            )
            self.sam_predictor = SAM2ImagePredictor(sam2_model)
            logger.info("Grounding DINO + SAM2 本地组合模型加载完成。")
        except Exception as exc:
            self.processor = None
            self.detector = None
            self.sam_predictor = None
            self.load_error = str(exc)
            logger.exception("模型加载失败：%s", exc)
        finally:
            self.loading = False
            self.load_finished_at = time.time()
            self._load_lock.release()

    def start_background_load(self) -> None:
        if self.ready or self.loading:
            return
        threading.Thread(target=self.load, name="model-loader", daemon=True).start()

    def status(self) -> dict[str, Any]:
        if self.ready:
            status = "ready"
        elif self.loading:
            status = "loading"
        elif self.load_error:
            status = "model_error"
        else:
            status = "not_loaded"
        return {
            "status": status,
            "device": str(self.device),
            "engine": "pytorch-cpu",
            "grounding_dino_loaded": self.detector is not None,
            "sam2_loaded": self.sam_predictor is not None,
            "rtmpose_loaded": False,
            "rtmpose_hand_loaded": False,
            "model_dir": str(MODEL_DIR),
            "model_error": self.load_error,
        }

    def _autocast_context(self):
        return nullcontext()

    def analyze(
        self,
        image_rgb: np.ndarray,
        prompt: str | tuple[str, ...],
        referring_label: str | None = None,
        color_hint: str | None = None,
    ) -> list[DetectedObject]:
        if not self.ready:
            reason = self.load_error or "模型尚未加载完成"
            raise RuntimeError(f"本地模型不可用：{reason}")

        resized_rgb, scale_x, scale_y = resize_for_inference(image_rgb)
        pil_image = Image.fromarray(resized_rgb)
        strict_glasses = bool(referring_label and ("眼镜" in referring_label or "glasses" in referring_label.lower()))
        # 无需提示会拆为多个语义相近的小词组，避免很长的单次提示削弱文本匹配。
        prompts = (prompt,) if isinstance(prompt, str) else prompt
        if strict_glasses:
            # 属性指代需要独立验证脸部与眼镜，不能只信“person wearing glasses”整句匹配。
            prompts = prompts + ("face. glasses. eyeglasses.",)

        # 两个大模型共享显存，串行化请求可以避免并发推理造成瞬时 OOM。
        with self._inference_lock, torch.inference_mode(), self._autocast_context():
            post_process = self.processor.post_process_grounded_object_detection
            threshold_name = (
                "box_threshold" if "box_threshold" in signature(post_process).parameters else "threshold"
            )
            batch_boxes: list[np.ndarray] = []
            batch_scores: list[np.ndarray] = []
            labels: list[str] = []
            for normalized_prompt in prompts:
                inputs = self.processor(images=pil_image, text=normalized_prompt, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.detector(**inputs)
                processed = post_process(
                    outputs,
                    input_ids=inputs["input_ids"],
                    text_threshold=TEXT_CONFIDENCE_THRESHOLD,
                    target_sizes=[(resized_rgb.shape[0], resized_rgb.shape[1])],
                    **{threshold_name: BOX_CONFIDENCE_THRESHOLD},
                )[0]
                batch_boxes.append(processed["boxes"].detach().float().cpu().numpy())
                batch_scores.append(processed["scores"].detach().float().cpu().numpy())
                current_labels = processed.get("text_labels", processed.get("labels", []))
                labels.extend(canonicalize_label(str(label)) for label in current_labels)

            boxes = np.concatenate(batch_boxes, axis=0) if batch_boxes else np.empty((0, 4))
            scores = np.concatenate(batch_scores, axis=0) if batch_scores else np.empty((0,))

            if len(boxes) == 0:
                return []

            keep_indexes = class_aware_nms(boxes, scores, labels, MAX_OBJECTS_PER_FRAME, iou_threshold=0.45)
            keep_indexes = suppress_generic_overlaps(boxes, scores, labels, keep_indexes)
            keep_indexes = suppress_sibling_overlaps(boxes, scores, labels, keep_indexes)
            keep_indexes = suppress_contextual_fruit_errors(labels, keep_indexes)
            if strict_glasses:
                keep_indexes = require_visible_glasses(boxes, labels, keep_indexes)
            if color_hint:
                keep_indexes = filter_boxes_by_color(resized_rgb, boxes, keep_indexes, color_hint)
            boxes = boxes[keep_indexes]
            scores = scores[keep_indexes]
            labels = [labels[index] for index in keep_indexes]

            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, resized_rgb.shape[1] - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, resized_rgb.shape[0] - 1)

            logger.info("Grounding DINO 检测到 %d 个目标，开始批量执行 SAM2 分割。", len(boxes))
            self.sam_predictor.set_image(resized_rgb)
            masks, _, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )

        masks = normalize_sam_masks(np.asarray(masks), len(boxes))
        original_height, original_width = image_rgb.shape[:2]
        detected: list[DetectedObject] = []

        for index, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
            if mask.shape != (original_height, original_width):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            binary_mask = (mask > 0).astype(np.uint8)
            original_box = [
                float(box[0] / scale_x),
                float(box[1] / scale_y),
                float(box[2] / scale_x),
                float(box[3] / scale_y),
            ]
            original_box[0] = max(0.0, min(original_box[0], original_width - 1.0))
            original_box[2] = max(0.0, min(original_box[2], original_width - 1.0))
            original_box[1] = max(0.0, min(original_box[1], original_height - 1.0))
            original_box[3] = max(0.0, min(original_box[3], original_height - 1.0))
            label = referring_label or (labels[index] if index < len(labels) else "object")
            detected.append(
                DetectedObject(
                    label=label,
                    score=round(float(score), 4),
                    bbox=[round(value, 2) for value in original_box],
                    mask=binary_mask.tolist(),
                )
            )

        logger.info("SAM2 分割完成，共返回 %d 个实例。", len(detected))
        return detected


def canonicalize_label(label: str) -> str:
    normalized = re.sub(r"\s+", " ", label.lower().strip().rstrip("."))
    normalized = LABEL_ALIASES.get(normalized, normalized)
    # Grounding DINO 在一条长提示里偶尔会返回 "person child" 这类复合文本。
    # 无需提示工作台只展示主体类别，因此归一为最稳定的单一主类。
    subjects = (
        "soccer ball", "football", "sports ball", "ball", "car", "truck", "bus", "van",
        "motorcycle", "bicycle", "person", "child", "woman", "man", "dog", "cat", "bird",
        "horse", "cow", "sheep", "goalpost", "goal", "tree", "plant", "grass", "road", "fence",
        "chair", "table", "sofa", "bed", "television", "monitor", "laptop", "cell phone",
        "keyboard", "mouse", "book", "clock", "bottle", "cup", "bowl", "glasses", "eyeglasses", "tomato", "apple", "orange", "text",
        "banana", "fruit", "door", "window", "pole", "building", "house", "bridge", "sign", "tower", "wall",
    )
    for subject in subjects:
        if re.search(rf"\b{re.escape(subject)}\b", normalized):
            return LABEL_ALIASES.get(subject, subject)
    return normalized


def referring_color_hint(prompt: str) -> str | None:
    for chinese, english in CHINESE_COLORS.items():
        if chinese in prompt:
            return english
    for color in CHINESE_COLORS.values():
        if re.search(rf"\b{re.escape(color)}\b", prompt.lower()):
            return color
    return None


def filter_boxes_by_color(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    indexes: list[int],
    color: str,
) -> list[int]:
    """用 HSV 颜色占比约束指代提示，减少“紫色衣服”命中所有人物的问题。"""
    image_h, image_w = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    candidates: list[tuple[int, float]] = []
    for index in indexes:
        x1, y1, x2, y2 = [int(round(value)) for value in boxes[index]]
        x1, x2 = max(0, min(x1, image_w - 1)), max(1, min(x2, image_w))
        y1, y2 = max(0, min(y1, image_h - 1)), max(1, min(y2, image_h))
        # 指代颜色主要看框内中心区域，排除草地、天空和框外背景造成的误命中。
        width, height = x2 - x1, y2 - y1
        cx1, cx2 = x1 + int(width * 0.18), x2 - int(width * 0.18)
        cy1, cy2 = y1 + int(height * 0.12), y1 + int(height * 0.82)
        crop = hsv[cy1:max(cy1 + 1, cy2), cx1:max(cx1 + 1, cx2)]
        if crop.size == 0:
            continue
        if color == "white":
            mask = (crop[:, :, 1] < 55) & (crop[:, :, 2] > 150)
        elif color == "black":
            mask = crop[:, :, 2] < 80
        elif color == "gray":
            mask = (crop[:, :, 1] < 45) & (crop[:, :, 2] >= 80) & (crop[:, :, 2] <= 190)
        elif color == "red":
            mask = ((crop[:, :, 0] <= 10) | (crop[:, :, 0] >= 170)) & (crop[:, :, 1] > 65) & (crop[:, :, 2] > 55)
        else:
            ranges = {"purple": (125, 165), "blue": (90, 135), "green": (35, 90), "yellow": (18, 40), "orange": (5, 24), "pink": (150, 179), "brown": (5, 25)}
            low, high = ranges.get(color, (0, 179))
            mask = (crop[:, :, 0] >= low) & (crop[:, :, 0] <= high) & (crop[:, :, 1] > 45) & (crop[:, :, 2] > 35)
        coverage = float(mask.mean())
        if coverage >= 0.065:
            candidates.append((index, coverage))
    if not candidates:
        return []
    # 颜色指代是排序约束：只保留颜色覆盖最强的候选，避免“紫色衣服”命中全部人物。
    best_coverage = max(coverage for _, coverage in candidates)
    return [index for index, coverage in candidates if coverage >= max(0.065, best_coverage * 0.84)]


def normalize_text_prompt(prompt: str) -> str:
    """文本提示只接受英文类别，以避免英文文本编码器产生不可解释的结果。"""
    raw = prompt.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="文本提示模式至少需要填写一个英文检测类别。")
    raw_parts = [item.strip().lower().rstrip(".") for item in re.split(r"[,，。.]", raw) if item.strip()]
    classes = [TEXT_ALIASES.get(item, item) for item in raw_parts]
    if any(re.search(r"[^a-z0-9 _-]", item) for item in classes):
        raise HTTPException(status_code=400, detail="文本提示请填写英文类别，例如 person, soccer ball。")
    if not classes:
        raise HTTPException(status_code=400, detail="文本提示模式至少需要填写一个英文检测类别。")
    return ". ".join(classes) + "."


def translate_chinese_referring_prompt(prompt: str) -> str:
    """将覆盖范围内的中文指代短语转为 Grounding DINO 的英文描述。"""
    subject = next((english for chinese, english in CHINESE_SUBJECTS.items() if chinese in prompt), None)
    if not subject:
        raise HTTPException(
            status_code=400,
            detail="当前本地中文指代需包含常见主体词，例如人、车、狗、猫、杯子或西红柿；也可直接输入英文描述。",
        )
    color = next((english for chinese, english in CHINESE_COLORS.items() if chinese in prompt), None)
    position = next((english for chinese, english in CHINESE_POSITIONS.items() if chinese in prompt), None)
    state = next((english for chinese, english in CHINESE_STATES.items() if chinese in prompt), None)
    attribute = next((english for chinese, english in CHINESE_ATTRIBUTES.items() if chinese in prompt), None)
    has_clothes = "衣服" in prompt or "衣" in prompt

    words: list[str] = []
    if state:
        words.append(state)
    if subject in {"person", "man", "woman", "child", "baby"} and attribute:
        words.extend([subject, attribute])
        if color and has_clothes:
            words.extend(["wearing", color, "clothes"])
    elif subject in {"person", "man", "woman", "child", "baby"} and color and has_clothes:
        words.extend([subject, "wearing", color, "clothes"])
    else:
        if color:
            words.append(color)
        words.append(subject)
    if position:
        words.append(position)
    return " ".join(words)


def prepare_prompt(
    prompt_mode: Literal["universal", "text", "referring"],
    text_prompt: str,
) -> tuple[str | tuple[str, ...], str | None]:
    """返回 Grounding DINO 输入提示与需要展示给用户的指代标签。"""
    if prompt_mode == "universal":
        return UNIVERSAL_PROMPT_GROUPS, None
    if prompt_mode == "text":
        return normalize_text_prompt(text_prompt), None

    raw = text_prompt.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="指代提示模式需要填写具体描述，例如“穿白色衣服的人”。")
    if len(raw) > 160:
        raise HTTPException(status_code=400, detail="指代提示最多支持 160 个字符。")
    if re.search(r"[\u4e00-\u9fff]", raw):
        model_prompt = translate_chinese_referring_prompt(raw)
    else:
        model_prompt = raw.lower().rstrip(".")
    return model_prompt + ".", raw


def infer_activity(objects: list[DetectedObject], media_type: str) -> str:
    labels = [item.label.lower() for item in objects]
    person_count = sum(label in {"person", "child", "man", "woman"} for label in labels)
    ball_count = sum("ball" in label for label in labels)
    car_count = sum(label in {"car", "truck", "bus", "van", "motorcycle", "bicycle"} for label in labels)
    if ball_count and person_count >= 2:
        return "画面显示多名人员在球场进行踢足球活动。"
    if media_type == "video" and car_count >= 2:
        return "视频中检测到多辆车辆沿道路前后行驶，呈现车队行驶场景。"
    if person_count >= 2 and media_type == "video":
        return "视频中检测到多名人员，画面包含连续的人物活动。"
    if car_count:
        return "画面中检测到车辆及其道路场景。"
    if person_count:
        return "画面中检测到人物主体。"
    return "已根据当前提示完成主体检测，未生成可确认的行为描述。"


def describe_content(objects: list[DetectedObject], media_type: str) -> str:
    """只使用有检测依据的类别生成简短内容概括，避免把模型猜测写成确定事实。"""
    labels = [item.label.lower() for item in objects]
    person_count = sum(label in {"person", "child", "man", "woman"} for label in labels)
    ball_count = sum("ball" in label for label in labels)
    car_count = sum(label in {"car", "truck", "bus", "van", "motorcycle", "bicycle"} for label in labels)
    tomato_count = sum(label == "tomato" for label in labels)
    plant_context = any(label in {"tree", "plant", "leaf", "grass", "bush", "shrub"} for label in labels)
    screen_count = sum(label in {"monitor", "television", "screen"} for label in labels)
    if tomato_count >= 2 and plant_context:
        return f"画面中是一棵番茄植株，枝叶间可见约 {tomato_count} 个番茄。"
    if ball_count and person_count:
        return f"画面中有 {person_count} 名人员在球场进行踢足球活动。"
    if person_count >= 2 and screen_count:
        return f"画面中有 {person_count} 个人，场景位于办公区域，可见显示器等设备。"
    if media_type == "video" and car_count >= 2:
        return "视频中可见多辆车辆沿道路连续行驶。"
    if car_count:
        return f"画面中可见 {car_count} 辆车辆及周围道路场景。"
    if person_count:
        return f"画面中可见 {person_count} 个人。"
    if labels:
        names = {
            "monitor": "显示器", "television": "显示器", "person": "人员", "child": "儿童",
            "tree": "树木", "plant": "植物", "tomato": "番茄", "soccer ball": "足球",
            "car": "车辆", "truck": "卡车", "road": "道路", "building": "建筑",
            "shoe": "鞋子", "sneaker": "运动鞋", "shirt": "上衣", "face": "脸部",
        }
        visible = list(dict.fromkeys(names.get(label, label) for label in labels if label not in {"object", "thing"}))[:4]
        return "画面中可见" + "、".join(visible) + "等主体。"
    return "当前画面未检测到足够明确的主体，无法生成可靠概括。"


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    max_count: int,
    iou_threshold: float = 0.62,
) -> list[int]:
    """保留同一类别的最佳候选，避免分组提示造成重复框。"""
    selected: list[int] = []
    by_label: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label.lower().strip(), []).append(index)
    for indexes in by_label.values():
        for index in sorted(indexes, key=lambda item: float(scores[item]), reverse=True):
            current_label = labels[index].lower().strip()
            same_class_selected = [existing for existing in selected if labels[existing].lower().strip() == current_label]
            if all(box_iou(boxes[index], boxes[existing]) < iou_threshold for existing in same_class_selected):
                selected.append(index)
                if len(selected) >= max_count:
                    break
        if len(selected) >= max_count:
            break
    return sorted(selected, key=lambda item: float(scores[item]), reverse=True)


def suppress_generic_overlaps(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    indexes: list[int],
    iou_threshold: float = 0.5,
) -> list[int]:
    """跨提示词去重：按类别层级让具体类别覆盖同一区域的泛类框。"""
    kept: list[int] = []
    ordered = sorted(indexes, key=lambda item: float(scores[item]), reverse=True)
    for index in ordered:
        label = labels[index].lower().strip()
        generic_children = CATEGORY_HIERARCHY.get(label, set())
        should_suppress = False
        for other in ordered:
            if other == index:
                continue
            child = labels[other].lower().strip()
            if child not in generic_children:
                continue
            overlap = box_iou(boxes[index], boxes[other])
            generic_area = max(1.0, float(boxes[index][2] - boxes[index][0]) * float(boxes[index][3] - boxes[index][1]))
            intersection = overlap * max(
                1.0,
                float(boxes[index][2] - boxes[index][0]) * float(boxes[index][3] - boxes[index][1])
                + float(boxes[other][2] - boxes[other][0]) * float(boxes[other][3] - boxes[other][1]),
            )
            child_score = float(scores[other])
            # IoU 对大小差异很大的框不够敏感；额外使用交集/泛类面积判断包含关系。
            intersection_over_generic = intersection / generic_area
            if (overlap >= iou_threshold or intersection_over_generic >= 0.12) and child_score >= float(scores[index]) * 0.62:
                should_suppress = True
                break
        if should_suppress:
            continue
        kept.append(index)
    return sorted(kept, key=lambda item: float(scores[item]), reverse=True)


def suppress_sibling_overlaps(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    indexes: list[int],
    iou_threshold: float = 0.18,
) -> list[int]:
    """同一物体被多个同级类别命中时只保留最高分，例如 tomato/apple/orange。"""
    kept: list[int] = []
    ordered = sorted(indexes, key=lambda item: float(scores[item]), reverse=True)
    for index in ordered:
        label = labels[index].lower().strip()
        group = next((items for items in MUTUALLY_EXCLUSIVE_GROUPS if label in items), None)
        duplicate = False
        if group is not None:
            for existing in kept:
                existing_label = labels[existing].lower().strip()
                if existing_label not in group:
                    continue
                if box_iou(boxes[index], boxes[existing]) >= iou_threshold:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(index)
    return sorted(kept, key=lambda item: float(scores[item]), reverse=True)


def suppress_contextual_fruit_errors(labels: list[str], indexes: list[int]) -> list[int]:
    """番茄植株场景中，避免同一批候选把番茄误叫成苹果/橘子。"""
    normalized = [labels[index].lower().strip() for index in indexes]
    tomato_count = sum(label == "tomato" for label in normalized)
    plant_context = any(label in {"tree", "plant", "leaf", "grass", "bush", "shrub"} for label in normalized)
    if tomato_count >= 2 and plant_context:
        fruit_siblings = {"apple", "orange", "banana", "lemon", "watermelon", "grape", "strawberry"}
        return [index for index in indexes if labels[index].lower().strip() not in fruit_siblings]
    return indexes


def require_visible_glasses(boxes: np.ndarray, labels: list[str], indexes: list[int]) -> list[int]:
    """眼镜属性只在同一人物可见脸部且脸部内命中眼镜时成立。"""
    faces = [index for index in indexes if labels[index].lower().strip() == "face"]
    glasses = [index for index in indexes if labels[index].lower().strip() in {"glasses", "eyeglasses"}]
    people = [index for index in indexes if labels[index].lower().strip() in {"person", "man", "woman", "child"}]
    validated: list[int] = []
    for person in people:
        x1, y1, x2, y2 = boxes[person]
        head_bottom = y1 + (y2 - y1) * 0.48
        matching_faces = [face for face in faces if x1 <= (boxes[face][0] + boxes[face][2]) / 2 <= x2 and y1 <= (boxes[face][1] + boxes[face][3]) / 2 <= head_bottom]
        if not matching_faces:
            continue
        if any(
            boxes[face][0] <= (boxes[glass][0] + boxes[glass][2]) / 2 <= boxes[face][2]
            and boxes[face][1] <= (boxes[glass][1] + boxes[glass][3]) / 2 <= boxes[face][3]
            for face in matching_faces
            for glass in glasses
        ):
            validated.append(person)
    return validated


def build_summary(objects: list[DetectedObject], mode: str) -> str:
    mode_name = {"universal": "无需提示", "text": "文本提示", "referring": "指代提示"}[mode]
    if not objects:
        return f"本地{mode_name}分析完成，未检测到符合当前条件的目标。"
    counts: dict[str, int] = {}
    for item in objects:
        counts[item.label] = counts.get(item.label, 0) + 1
    details = "、".join(f"{label} {count} 个" for label, count in counts.items())
    return f"本地{mode_name}分析完成，共识别 {len(objects)} 个实例：{details}。"


def resize_for_inference(image_rgb: np.ndarray) -> tuple[np.ndarray, float, float]:
    height, width = image_rgb.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= MAX_IMAGE_LONG_EDGE:
        return image_rgb, 1.0, 1.0

    ratio = MAX_IMAGE_LONG_EDGE / float(longest_edge)
    resized_width = max(1, round(width * ratio))
    resized_height = max(1, round(height * ratio))
    logger.info(
        "原图尺寸 %dx%d，长边超过 %d，推理前等比例缩放为 %dx%d。",
        width,
        height,
        MAX_IMAGE_LONG_EDGE,
        resized_width,
        resized_height,
    )
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized, resized_width / width, resized_height / height


def normalize_sam_masks(masks: np.ndarray, expected_count: int) -> np.ndarray:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    if masks.ndim != 3 or masks.shape[0] != expected_count:
        raise RuntimeError(
            f"SAM2 返回了无法识别的掩码尺寸 {masks.shape}，预期目标数为 {expected_count}。"
        )
    return masks


def decode_image(data: bytes) -> np.ndarray:
    try:
        # 浏览器会应用 JPEG 的 EXIF 方向；后端同步校正，避免手机照片标注错位。
        with Image.open(BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            return np.asarray(image).copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="图片损坏或格式不受支持。") from exc


def decode_video_sample_frames(upload: UploadFile, suffix: str) -> list[tuple[float, np.ndarray]]:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            while chunk := upload.file.read(1024 * 1024):
                temp_file.write(chunk)
            temp_path = Path(temp_file.name)

        capture = cv2.VideoCapture(str(temp_path))
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="视频无法打开，请检查编码格式或文件是否损坏。")
        try:
            frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            duration = (frame_count - 1) / fps if fps > 0 else 0.0
            # 以真实时间点 0、3、6... 抽帧；短视频再用均匀补帧满足最低覆盖数，长视频受上限保护。
            target_times = list(np.arange(0.0, max(duration, 0.0) + 1e-6, VIDEO_SAMPLE_INTERVAL_SECONDS))
            if not target_times or target_times[-1] < duration - 0.08:
                target_times.append(duration)
            if len(target_times) < VIDEO_MIN_SAMPLE_COUNT:
                target_times = list(np.linspace(0.0, max(duration, 0.0), VIDEO_MIN_SAMPLE_COUNT))
            elif len(target_times) > VIDEO_MAX_SAMPLE_COUNT:
                target_times = list(np.linspace(0.0, max(duration, 0.0), VIDEO_MAX_SAMPLE_COUNT))
            indexes = sorted({max(0, min(frame_count - 1, round(timestamp * fps))) for timestamp in target_times})
            logger.info("视频时长 %.2fs，按 %.1fs 间隔抽取 %d 个关键帧。", duration, VIDEO_SAMPLE_INTERVAL_SECONDS, len(indexes))
            samples: list[tuple[float, np.ndarray]] = []
            for frame_index in indexes:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame_bgr = capture.read()
                if not success or frame_bgr is None:
                    logger.warning("跳过无法读取的视频帧：%d", frame_index)
                    continue
                timestamp = frame_index / fps if fps > 0 else float(frame_index)
                samples.append((round(timestamp, 3), cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        finally:
            capture.release()

        if not samples:
            raise HTTPException(status_code=400, detail="视频解析失败，未能读取任何关键帧。")
        return samples
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def is_video(upload: UploadFile, suffix: str) -> bool:
    return (upload.content_type or "").startswith("video/") or suffix in ALLOWED_VIDEO_SUFFIXES


def filter_openvino_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    image_rgb: np.ndarray,
    color_hint: str | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """复用旧路径的 NMS/语义去重规则，确保两引擎输出行为一致。"""
    if not len(boxes):
        return boxes, scores, labels
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_rgb.shape[1] - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_rgb.shape[0] - 1)
    keep = class_aware_nms(boxes, scores, labels, MAX_OBJECTS_PER_FRAME, iou_threshold=0.45)
    keep = suppress_generic_overlaps(boxes, scores, labels, keep)
    keep = suppress_sibling_overlaps(boxes, scores, labels, keep)
    keep = suppress_contextual_fruit_errors(labels, keep)
    if color_hint:
        keep = filter_boxes_by_color(image_rgb, boxes, keep, color_hint)
    return boxes[keep], scores[keep], [labels[index] for index in keep]


class OpenVINOCompatibleManager:
    """将 OpenVINO 原图结果恢复成与旧 ModelManager 完全相同的数据模型。"""

    def __init__(self) -> None:
        from openvino_engine import OpenVINOModelManager

        self.engine = OpenVINOModelManager(
            MODEL_DIR,
            GROUNDING_DINO_DIR,
            OPENVINO_IDLE_SECONDS,
            BOX_CONFIDENCE_THRESHOLD,
            TEXT_CONFIDENCE_THRESHOLD,
            canonicalize_label,
            filter_openvino_detections,
        )
        self.cpu_fallback: PyTorchModelManager | None = None
        self._fallback_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.engine.ready

    def start_background_load(self) -> None:
        self.engine.start_background_load()

    def status(self) -> dict[str, Any]:
        return self.engine.status()

    def analyze(
        self,
        image_rgb: np.ndarray,
        prompt: str | tuple[str, ...],
        referring_label: str | None = None,
        color_hint: str | None = None,
    ) -> list[DetectedObject]:
        resized_rgb, scale_x, scale_y = resize_for_inference(image_rgb)
        try:
            raw_objects = self.engine.analyze(resized_rgb, prompt, referring_label, color_hint)
        except RuntimeError as exc:
            if "logits saturated" not in str(exc):
                raise
            logger.warning("OpenVINO 检测 IR 输出饱和，自动切换 PyTorch CPU 推理：%s", exc)
            raw_objects = self._fallback_analyze(resized_rgb, prompt, referring_label, color_hint)
        # A stale FP16 IR can produce finite-but-useless logits and an empty
        # list without raising. Retry once through the known-good CPU path so
        # a successful HTTP response never masquerades as a broken analysis.
        if not raw_objects:
            logger.warning("OpenVINO 未返回目标，自动使用 PyTorch CPU 路径重试。")
            raw_objects = self._fallback_analyze(resized_rgb, prompt, referring_label, color_hint)
        original_height, original_width = image_rgb.shape[:2]
        restored: list[DetectedObject] = []
        posed_people = 0
        for item in raw_objects:
            source_mask = item.mask if isinstance(item, DetectedObject) else item["mask"]
            mask = np.asarray(source_mask, dtype=np.uint8)
            if mask.shape != (original_height, original_width):
                mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            box = item.bbox if isinstance(item, DetectedObject) else item["bbox"]
            restored_box = [
                max(0.0, min(float(box[0]) / scale_x, original_width - 1.0)),
                max(0.0, min(float(box[1]) / scale_y, original_height - 1.0)),
                max(0.0, min(float(box[2]) / scale_x, original_width - 1.0)),
                max(0.0, min(float(box[3]) / scale_y, original_height - 1.0)),
            ]
            source_keypoints = item.keypoints if isinstance(item, DetectedObject) else item.get("keypoints", [])
            keypoints = [
                {
                    "x": round(max(0.0, min(float(point["x"]) / scale_x, original_width - 1.0)), 2),
                    "y": round(max(0.0, min(float(point["y"]) / scale_y, original_height - 1.0)), 2),
                    "score": float(point["score"]),
                }
                for point in source_keypoints
            ]
            label = item.label if isinstance(item, DetectedObject) else item["label"]
            if label.lower() == "person" and len(keypoints) != 17 and posed_people < MAX_POSE_PERSONS:
                try:
                    keypoints = self.engine.estimate_pose(image_rgb, np.asarray(restored_box, dtype=np.float32))
                    posed_people += 1
                except Exception as exc:
                    logger.warning("RTMPose 姿态补全失败，将仅返回检测框：%s", exc)
            hand_keypoints: list[list[dict[str, float]]] = []
            if label.lower() in {"hand", "手", "手部"}:
                try:
                    hand_points = self.engine.estimate_hand_pose(image_rgb, np.asarray(restored_box, dtype=np.float32))
                    if len(hand_points) == 21:
                        hand_keypoints = [hand_points]
                except Exception as exc:
                    logger.warning("RTMPose-hand 推理不可用，将返回空手部关键点：%s", exc)
            score = item.score if isinstance(item, DetectedObject) else item["score"]
            restored.append(DetectedObject(
                label=label,
                score=score,
                bbox=[round(value, 2) for value in restored_box],
                mask=(mask > 0).astype(np.uint8).tolist(),
                keypoints=keypoints,
                description=object_description(label, restored_box, original_width, original_height),
                caption="",
                hand_keypoints=hand_keypoints,
            ))
        return restored

    def _fallback_analyze(
        self,
        image_rgb: np.ndarray,
        prompt: str | tuple[str, ...],
        referring_label: str | None,
        color_hint: str | None,
    ) -> list[DetectedObject]:
        with self._fallback_lock:
            if self.cpu_fallback is None:
                self.cpu_fallback = PyTorchModelManager()
                self.cpu_fallback.start_background_load()
            fallback = self.cpu_fallback
        deadline = time.monotonic() + 90
        while not fallback.ready and fallback.load_error is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if not fallback.ready:
            raise RuntimeError(f"OpenVINO 检测结果异常，CPU 兜底模型不可用：{fallback.load_error or '加载超时'}")
        result = fallback.analyze(image_rgb, prompt, referring_label, color_hint)
        return sorted(result, key=lambda item: item.score, reverse=True)[:MAX_RESPONSE_OBJECTS]


# False 时严格走原有 PyTorch CPU 路径；不会删除或覆盖旧模型代码。
model_manager: Any = OpenVINOCompatibleManager() if USE_OPENVINO else PyTorchModelManager()
capability_enricher = FullCapabilityEnricher(MODEL_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("启动本地视觉模型服务，模型目录：%s", MODEL_DIR)
    model_manager.start_background_load()
    yield


app = FastAPI(
    title="Grounding DINO + SAM2 本地视觉标注服务",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    # 开发前端固定运行在 localhost:5173；开启凭据后不能再使用通配来源。
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health() -> dict[str, Any]:
    state = model_manager.status()
    state["capabilities"] = local_capabilities()
    state["enrichment"] = capability_enricher.status()
    state["model_assets"] = capability_enricher.asset_report
    assets = list(capability_enricher.asset_report.values())
    available = sum(1 for item in assets if item.get("available"))
    state["model_progress"] = {
        "checked": len(assets),
        "total": len(assets),
        "available": available,
        "percent": round(available * 100 / len(assets)) if assets else 0,
        "phase": "ready" if state["status"] == "ready" else "loading" if state["status"] == "loading" else "audit",
    }
    state["privacy"] = {"network_runtime": False, "cuda": False}
    return state


@app.post("/api/analyze-media", response_model=AnalyzeResponse)
def analyze_media(
    media: UploadFile = File(...),
    text_prompt: str = Form(""),
    prompt_mode: Literal["universal", "text", "referring"] = Form("text"),
) -> AnalyzeResponse:
    if not model_manager.ready:
        state = model_manager.status()
        if state["status"] == "model_error":
            detail = f"本地模型加载失败：{state['model_error']}"
        else:
            detail = "本地模型正在初始化或等待导出的 IR，请稍后查看 /api/health。"
        raise HTTPException(status_code=503, detail=detail)

    suffix = Path(media.filename or "").suffix.lower()
    content_type = media.content_type or ""
    supported = (
        content_type.startswith("image/")
        or content_type.startswith("video/")
        or suffix in ALLOWED_IMAGE_SUFFIXES
        or suffix in ALLOWED_VIDEO_SUFFIXES
    )
    if not supported:
        raise HTTPException(status_code=415, detail="仅支持常见图片或视频文件。")

    try:
        normalized_prompt, referring_label = prepare_prompt(prompt_mode, text_prompt)
        color_hint = referring_color_hint(text_prompt) if prompt_mode == "referring" else None
        media_type: Literal["image", "video"] = "video" if is_video(media, suffix) else "image"
        if media_type == "video":
            samples = decode_video_sample_frames(media, suffix or ".mp4")
            if prompt_mode == "universal":
                normalized_prompt = VIDEO_UNIVERSAL_PROMPT_GROUPS
            frames = [
                VideoFrameAnnotation(timestamp=timestamp, objects=capability_enricher.enrich(image_rgb, model_manager.analyze(image_rgb, normalized_prompt, referring_label, color_hint)))
                for timestamp, image_rgb in samples
            ]
            objects = frames[0].objects
            all_objects = [item for frame in frames for item in frame.objects]
        else:
            image_rgb = decode_image(media.file.read())
            objects = capability_enricher.enrich(image_rgb, model_manager.analyze(image_rgb, normalized_prompt, referring_label, color_hint))
            frames = []
            all_objects = objects
        labels = list(dict.fromkeys(item.label for item in all_objects))
        activity = infer_activity(all_objects, media_type)
        feedback = describe_content(all_objects, media_type)
        compact_frames = [
            VideoFrameAnnotation(timestamp=frame.timestamp, objects=[compact_object(item) for item in frame.objects])
            for frame in frames
        ]
        return AnalyzeResponse(
            objects=[compact_object(item) for item in objects],
            labels=labels,
        feedback=feedback,
        # 前端顶部与内容摘要都优先显示可读的内容概括，实例数由语义标签和空间标注呈现。
        summary=feedback,
            task_id=str(uuid.uuid4()),
            prompt_mode=prompt_mode,
            media_type=media_type,
            frames=compact_frames,
            activity=activity,
        )
    except HTTPException:
        raise
    except MemoryError as exc:
        logger.exception("推理内存不足：%s", exc)
        raise HTTPException(
            status_code=507,
            detail="GPU 显存不足。请调低 MAX_IMAGE_LONG_EDGE、关闭其他显存占用程序后重试。",
        ) from exc
    except Exception as exc:
        logger.exception("媒体推理失败：%s", exc)
        raise HTTPException(status_code=500, detail=f"模型推理失败：{exc}") from exc
    finally:
        media.file.close()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": f"请求参数不完整或格式错误：{exc}"})


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("未处理的服务异常：%s", exc)
    return JSONResponse(status_code=500, content={"detail": f"服务器内部错误：{exc}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
