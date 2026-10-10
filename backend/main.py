from __future__ import annotations

import logging
import json
import math
import os
import ctypes
import re
import tempfile
import threading
import time
import tracemalloc
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
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
from recall_scheduler import Candidate, adaptive_soft_nms, conservative_semantic_deduplicate, validate_high_risk_candidates

# ======================== 可配置参数 ========================
USE_OPENVINO = os.getenv("USE_OPENVINO", "1") not in {"0", "false", "False"}
BOX_CONFIDENCE_THRESHOLD = float(os.getenv("BOX_CONFIDENCE_THRESHOLD", "0.25"))
TEXT_CONFIDENCE_THRESHOLD = float(os.getenv("TEXT_CONFIDENCE_THRESHOLD", "0.22"))
MAX_IMAGE_LONG_EDGE = int(os.getenv("MAX_IMAGE_LONG_EDGE", "1280"))
OPENVINO_IDLE_SECONDS = int(os.getenv("OPENVINO_IDLE_SECONDS", "90"))
VIDEO_SAMPLE_INTERVAL_SECONDS = 3.0  # 视频每隔约 3 秒取一个关键帧
VIDEO_MIN_SAMPLE_COUNT = 4
VIDEO_MAX_SAMPLE_COUNT = 18
# // 优化：视频关键帧 - 低成本画面变化扫描只决定哪些帧进入重型检测。
VIDEO_SCENE_SCAN_INTERVAL_SECONDS = float(os.getenv("VIDEO_SCENE_SCAN_INTERVAL_SECONDS", "0.75"))
VIDEO_SCENE_CHANGE_THRESHOLD = float(os.getenv("VIDEO_SCENE_CHANGE_THRESHOLD", "0.28"))
MAX_OBJECTS_PER_FRAME = int(os.getenv("MAX_OBJECTS_PER_FRAME", "100"))
# 检测框和 SAM2 掩码是两种不同成本。所有通过 NMS 的真实检测都会返回，
# 只有最高分的一部分进入分割，避免街景等场景因几十个框进入分钟级等待。
MAX_RESPONSE_OBJECTS = int(os.getenv("MAX_RESPONSE_OBJECTS", "80"))
# SAM2 CPU 兜底同样逐框/批量计算；限制分割实例数避免多目标图片进入分钟级。
MAX_SEGMENTED_OBJECTS = max(1, int(os.getenv("MAX_SEGMENTED_OBJECTS", "24")))
# GroundingDINO 文本编码器最多 256 token。每组保持短小、按常见场景拆分，
# 默认跑完四组经过场景审计的核心类别。部署者可显式设为 5+，再加入
# universal_categories.txt 的扩展组；这样不会让旧词表中的服饰等噪声拖慢默认检测。
MAX_UNIVERSAL_PROMPT_GROUPS = max(1, int(os.getenv("MAX_UNIVERSAL_PROMPT_GROUPS", "4")))
MAX_POSE_PERSONS = int(os.getenv("MAX_POSE_PERSONS", "12"))
# // 优化：双运行模式 - fast 默认只在触发时切片；full 保证执行局部补检但不扩大 SAM2 配额。
DEFAULT_RUN_MODE = os.getenv("DEFAULT_RUN_MODE", "fast").lower()
if DEFAULT_RUN_MODE not in {"fast", "full"}:
    DEFAULT_RUN_MODE = "fast"
ENABLE_STAGED_RECALL = os.getenv("ENABLE_STAGED_RECALL", "1") not in {"0", "false", "False"}
ENABLE_TILE_RECALL = os.getenv("ENABLE_TILE_RECALL", "1") not in {"0", "false", "False"}
RAW_CANDIDATE_LIMIT = max(1, int(os.getenv("RAW_CANDIDATE_LIMIT", "240")))

# 无需提示模式使用本地 Grounding DINO 的常见主体类别词表。Grounding DINO 本身需要
# 文本输入，因此这不是云端“万物模型”，而是可审计、可按业务扩充的本地候选集合。
BASE_UNIVERSAL_PROMPT_GROUPS = (
    # 不把 head/face/arm 等人体局部与完整人物混在无提示词表中，减少重复框。
    "person. child. man. woman. pedestrian. cyclist. dog. cat. bird. horse. cow. sheep. backpack. handbag. umbrella. helmet",
    # 明确加入道路车辆、井盖和道路设施；不含 goal/head 等容易诱发错类的词。
    "car. sedan. automobile. vehicle. truck. pickup truck. SUV. van. bus. motorcycle. bicycle. train. boat. airplane. tire. wheel. road. dirt road. sidewalk. crosswalk. parking lot. manhole cover. guardrail. traffic light. traffic sign. street lamp. fire hydrant. trolley. shopping cart. trash can. bin",
    # 番茄植株、常见果蔬及建筑环境独立成组，避免车辆词表挤占 token 预算。
    "tomato. tomatoes. ripe tomato. red tomato. tomato fruit. tomato plant. tomato vine. cluster of tomatoes. apple. orange. banana. lemon. fruit. plant. tree. leaf. grass. flower. bush. shrub. field. desert. sand. mountain. hill. fence. building. house. bridge. sign. billboard. storefront. warehouse. school. hospital. apartment",
    # 家居、电器及常见物品；把空调外机作为独立概念明确提示。
    "air conditioner. outdoor air conditioner. AC outdoor unit. HVAC unit. chair. table. sofa. bed. refrigerator. oven. sink. toilet. bathtub. television. monitor. laptop. cell phone. keyboard. mouse. book. clock. door. window. cabinet. counter. bottle. cup. bowl. plate. shoe",
)

# 视频优先车辆和地表/道路，人体与动物放第三组；OCR 独立识别，不把 text 当物体候选。
VIDEO_UNIVERSAL_PROMPT_GROUPS = (
    "car. sedan. automobile. vehicle. truck. pickup truck. SUV. van. bus. motorcycle. bicycle. off-road vehicle. convoy of vehicles. wheel. tire. road. dirt road. rural road. tire tracks. guardrail. bridge. manhole cover",
    "desert. field. sand. mountain. hill. grass. tree. plant. bush. shrub. road sign. street lamp. building. house. fence. road. sky. cloud",
    "person. pedestrian. cyclist. dog. cat. horse. sheep. cow. backpack. umbrella",
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
    "sedan": "car", "off-road vehicle": "car", "tomato plant": "tomato plant",
    "tomatoes": "tomato", "ripe tomato": "tomato", "red tomato": "tomato", "tomato fruit": "tomato",
    "tomato vine": "tomato plant", "air conditioner outdoor unit": "air conditioner",
    "outdoor air conditioner": "air conditioner", "ac outdoor unit": "air conditioner", "hvac unit": "air conditioner",
    "fruit": "fruit",
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
WEIGHTS_DIR = Path(os.getenv("MODEL_WEIGHTS_DIR", str(BASE_DIR.parent / "weights"))).resolve()
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
        blocked = {
            "food", "animal", "plant", "vegetation", "fruit", "building", "vehicle", "object", "thing", "item", "stuff", "equipment", "device",
            # OCR 单独负责 text；这些人体局部/运动场地概念在开放候选词表里
            # 容易把整个人或无关背景切成碎片。需要时仍可用显式文本提示检测。
            "text", "head", "face", "arm", "arms", "leg", "legs", "shorts", "goal", "goalpost", "goal post", "vest", "clothing",
        }
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
    # GroundingDINO 的文本编码器最多接受 256 token。按约 40 个短类别切组，
    # 让每组都完整参与匹配，不依赖处理器悄悄截断词表后半段。
    chunks = [". ".join(merged[index : index + 40]) for index in range(0, len(merged), 40)]
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
    track_id: str = ""


class RawCandidate(BaseModel):
    """Debug-only candidate trace: distinguishes model misses from suppression."""

    label: str
    score: float
    bbox: list[float]
    stage: int = 1
    tile_id: str = "full"


def local_capabilities() -> dict[str, dict[str, Any]]:
    """Report the real runtime for each capability, not just XML file presence."""
    ir_dir = MODEL_DIR / "openvino"
    files = {
        "grounding_dino": ir_dir / "grounding_dino.xml",
        "sam2_small": ir_dir / "sam2_encoder.xml",
        "rtmpose_tiny": ir_dir / "rtmpose_tiny.xml",
        "rtmpose_hand": ir_dir / "rtmpose_hand.xml",
        "paddleocr": ir_dir / "paddleocr.xml",
        # Dedicated Moondream2 exporter is a multi-file Stateful bundle.
        "moondream2": ir_dir / "moondream2" / "decoder.xml",
    }
    capabilities = {
        name: {"available": path.exists() and path.with_suffix('.bin').exists(), "path": str(path), "engine": "openvino", "openvino_ir": path.exists() and path.with_suffix('.bin').exists()}
        for name, path in files.items()
    }
    hand_tflite = WEIGHTS_DIR / "rtmpose-hand-litert" / "rtmhand_fp16.tflite"
    if not capabilities["rtmpose_hand"]["available"] and hand_tflite.is_file():
        capabilities["rtmpose_hand"] = {"available": True, "path": str(hand_tflite), "engine": "litert-local", "openvino_ir": False}
    moondream_weights = WEIGHTS_DIR / "moondream2" / "model.safetensors"
    if not capabilities["moondream2"]["available"] and moondream_weights.is_file():
        capabilities["moondream2"] = {
            "available": False,
            "asset_present": True,
            "path": str(moondream_weights),
            "engine": "pytorch-cpu",
            "openvino_ir": False,
            "error": "本地 safetensors 完整，但 Moondream2 OpenVINO IR 尚未导出；详见 enrichment.caption",
        }
    return capabilities


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
    """给出简洁位置与类别描述；不把框面积伪装成语义说明。"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    horizontal = "左侧" if center_x < image_width / 3 else "右侧" if center_x > image_width * 2 / 3 else "中部"
    vertical = "上方" if center_y < image_height / 3 else "下方" if center_y > image_height * 2 / 3 else "中部"
    location = horizontal if horizontal == "中部" else f"{horizontal}{vertical}"
    if label.lower() == "text":
        return f"画面{location}的文字区域"
    return f"画面{location}的{label}"


def _finite_number(value: Any) -> float | None:
    """将 numpy/Python 数值转换为 JSON 可用的有限浮点数。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def process_memory_usage_mb() -> tuple[float, float]:
    """Return current and peak process working set without adding a psutil dependency."""
    # // 优化：资源监控 - Windows 工作集反映真实进程内存，不只统计 Python 堆分配。
    if os.name != "nt":
        return 0.0, 0.0
    try:
        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        success = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb,  # type: ignore[attr-defined]
        )
        if success:
            return round(counters.WorkingSetSize / 1024 / 1024, 2), round(counters.PeakWorkingSetSize / 1024 / 1024, 2)
    except Exception:
        pass
    return 0.0, 0.0


def _compact_keypoints(points: Any) -> list[dict[str, float]]:
    """只保留模型实际返回且坐标、置信度均为有限值的关键点。"""
    cleaned: list[dict[str, float]] = []
    for point in points if isinstance(points, list) else []:
        if not isinstance(point, dict):
            continue
        x, y, score = (_finite_number(point.get(key)) for key in ("x", "y", "score"))
        if x is None or y is None or score is None:
            continue
        cleaned.append({"x": round(x, 2), "y": round(y, 2), "score": round(score, 4)})
    return cleaned


def compact_object(item: DetectedObject) -> DetectedObject | None:
    """压缩掩码并拦截模型输出中的 NaN/Infinity。

    OpenVINO 异常输出不能被 JSON 编码。检测框或置信度异常时该实例没有
    可用的真实标注信息，因此丢弃该实例而不是用伪造数值替代。
    """
    bbox = [_finite_number(value) for value in item.bbox]
    score = _finite_number(item.score)
    if len(bbox) != 4 or any(value is None for value in bbox) or score is None:
        logger.warning("跳过包含非有限坐标或置信度的检测实例：%s", item.label)
        return None
    raw_mask = item.mask if isinstance(item.mask, list) else []
    try:
        mask_rle, mask_size = encode_mask_rle(raw_mask)
    except (TypeError, ValueError):
        # 掩码异常不影响仍然有效的真实 bbox 检测结果。
        mask_rle, mask_size = "", []
        logger.warning("检测实例 %s 的掩码格式异常，已省略掩码。", item.label)
    hand_keypoints = [_compact_keypoints(hand) for hand in item.hand_keypoints if isinstance(hand, list)]
    hand_keypoints = [hand for hand in hand_keypoints if hand]
    return DetectedObject(
        label=str(item.label),
        score=round(score, 4),
        bbox=[round(value, 2) for value in bbox if value is not None],
        mask=mask_rle,
        mask_rle=mask_rle,
        mask_size=mask_size,
        keypoints=_compact_keypoints(item.keypoints),
        hand_keypoints=hand_keypoints,
        ocr_text=str(item.ocr_text or ""),
        description=str(item.description or ""),
        caption=str(item.caption or ""),
        track_id=str(item.track_id or ""),
    )


def compact_objects(items: list[DetectedObject]) -> list[DetectedObject]:
    """批量压缩并过滤无法安全序列化的模型实例。"""
    return [result for item in items if (result := compact_object(item)) is not None]


class VideoFrameAnnotation(BaseModel):
    timestamp: float
    objects: list[DetectedObject]
    is_keyframe: bool = True


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
    run_mode: Literal["fast", "full"] = "fast"
    raw_candidates: list[RawCandidate] = []
    timings_ms: dict[str, float] = {}
    resource_usage: dict[str, float] = {}


class SegmentResponse(BaseModel):
    bbox: list[float]
    mask_rle: str
    mask_size: list[int]


class CaptionTaskResponse(BaseModel):
    task_id: str
    status: Literal["queued", "running", "completed", "failed", "unavailable"]
    caption: str = ""
    error: str = ""


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
        self.last_diagnostics: dict[str, Any] = {"raw_candidates": [], "timings_ms": {}}

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

    def segment_box(self, image_rgb: np.ndarray, bbox: list[float]) -> np.ndarray:
        """Run SAM2 for one user-selected original-image box only."""
        if not self.ready:
            raise RuntimeError(self.load_error or "本地模型尚未加载完成")
        resized, scale_x, scale_y = resize_for_inference(image_rgb)
        box = np.asarray([[bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]], dtype=np.float32)
        with self._inference_lock, torch.inference_mode():
            self.sam_predictor.set_image(resized)
            masks, _, _ = self.sam_predictor.predict(point_coords=None, point_labels=None, box=box, multimask_output=False)
        mask = normalize_sam_masks(np.asarray(masks), 1)[0]
        if mask.shape != image_rgb.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8)

    def analyze(
        self,
        image_rgb: np.ndarray,
        prompt: str | tuple[str, ...],
        referring_label: str | None = None,
        color_hint: str | None = None,
        run_mode: Literal["fast", "full"] = "fast",
        lightweight: bool = False,
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
                # GroundingDINO 的文本编码器固定支持最多 256 个 token。
                # 长类别列表必须截断到模型上限，否则会在 forward 阶段触发
                # ``size of tensor ... must match ... 256`` 并让整次分析返回 500。
                inputs = self.processor(
                    images=pil_image,
                    text=normalized_prompt,
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
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

            raw_candidates = [
                {"bbox": [float(value) for value in box], "score": float(score), "label": label, "stage": 1, "tile_id": "full"}
                for box, score, label in zip(boxes, scores, labels)
            ]
            if len(boxes) == 0:
                self.last_diagnostics = {"raw_candidates": raw_candidates, "timings_ms": {"dino_detection": 0.0, "nms_postprocess": 0.0, "sam2_segmentation": 0.0}}
                return []

            # // 优化：CPU 回退一致性 - 不再应用会删掉相邻小目标的泛类/果实硬过滤。
            keep_indexes = class_aware_nms(boxes, scores, labels, MAX_OBJECTS_PER_FRAME, iou_threshold=0.45)
            # // 优化：类别真实性校验 - CPU 兜底路径使用与 OpenVINO 相同的高风险类校验。
            keep_indexes = keep_validated_high_risk_indexes(resized_rgb, boxes, scores, labels, keep_indexes)
            if strict_glasses:
                keep_indexes = require_visible_glasses(boxes, labels, keep_indexes)
            if color_hint:
                keep_indexes = filter_boxes_by_color(resized_rgb, boxes, keep_indexes, color_hint)
            if not color_hint:
                keep_indexes = sorted(keep_indexes, key=lambda index: float(scores[index]), reverse=True)
            keep_indexes = keep_indexes[:MAX_RESPONSE_OBJECTS]
            segment_indexes = keep_indexes[:MAX_SEGMENTED_OBJECTS]
            if len(keep_indexes) > MAX_SEGMENTED_OBJECTS:
                logger.info(
                    "CPU 兜底返回 %d 个检测实例，其中得分最高的 %d 个执行 SAM2 分割。",
                    len(keep_indexes),
                    MAX_SEGMENTED_OBJECTS,
                )
            selected_boxes = boxes[keep_indexes]
            selected_scores = scores[keep_indexes]
            selected_labels = [labels[index] for index in keep_indexes]
            segment_positions = {source: position for position, source in enumerate(segment_indexes)}
            segment_boxes = boxes[segment_indexes]

            selected_boxes[:, [0, 2]] = np.clip(selected_boxes[:, [0, 2]], 0, resized_rgb.shape[1] - 1)
            selected_boxes[:, [1, 3]] = np.clip(selected_boxes[:, [1, 3]], 0, resized_rgb.shape[0] - 1)
            segment_boxes[:, [0, 2]] = np.clip(segment_boxes[:, [0, 2]], 0, resized_rgb.shape[1] - 1)
            segment_boxes[:, [1, 3]] = np.clip(segment_boxes[:, [1, 3]], 0, resized_rgb.shape[0] - 1)

            logger.info("Grounding DINO 检测到 %d 个目标，开始为 %d 个高分目标执行 SAM2 分割。", len(selected_boxes), len(segment_boxes))
            self.sam_predictor.set_image(resized_rgb)
            # 当前 SAM2 checkpoint 在大批量 box decoder 时可能出现 batch
            # 维度断言。按 8 框分批，失败时只放弃掩码，不丢弃真实检测框。
            mask_batches: list[np.ndarray] = []
            try:
                for start in range(0, len(segment_boxes), 8):
                    batch = segment_boxes[start : start + 8]
                    batch_masks, _, _ = self.sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=batch,
                        multimask_output=False,
                    )
                    mask_batches.append(np.asarray(batch_masks))
                masks = np.concatenate(mask_batches, axis=0) if mask_batches else np.empty((0, 0, 0), dtype=np.uint8)
            except Exception as exc:
                logger.warning("SAM2 分批分割失败，将保留检测框并省略掩码：%s", exc)
                masks = np.empty((0, 0, 0), dtype=np.uint8)

        masks = normalize_sam_masks(np.asarray(masks), len(segment_boxes))
        original_height, original_width = image_rgb.shape[:2]
        detected: list[DetectedObject] = []

        for index, (box, score, label) in enumerate(zip(selected_boxes, selected_scores, selected_labels)):
            source_index = keep_indexes[index]
            mask_position = segment_positions.get(source_index)
            mask = masks[mask_position] if mask_position is not None else None
            if mask is not None and mask.size and mask.shape != (original_height, original_width):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            binary_mask = (mask > 0).astype(np.uint8) if mask is not None and mask.size else None
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
            detected.append(
                DetectedObject(
                    label=referring_label or label,
                    score=round(float(score), 4),
                    bbox=[round(value, 2) for value in original_box],
                    # 空掩码明确表示该低优先级实例未进入分割批次，不伪造分割结果。
                    mask=binary_mask.tolist() if binary_mask is not None else "",
                )
            )

        logger.info("SAM2 分割完成，返回 %d 个检测实例，其中 %d 个带真实掩码。", len(detected), len(segment_boxes))
        self.last_diagnostics = {"raw_candidates": raw_candidates, "timings_ms": {}, "fallback": "pytorch"}
        return detected


def canonicalize_label(label: str) -> str:
    normalized = re.sub(r"\s+", " ", label.lower().strip().rstrip("."))
    normalized = LABEL_ALIASES.get(normalized, normalized)
    # Grounding DINO 在一条长提示里偶尔会返回 "person child" 这类复合文本。
    # 无需提示工作台只展示主体类别，因此归一为最稳定的单一主类。
    subjects = (
        "outdoor air conditioner", "air conditioner", "ac outdoor unit", "hvac unit", "manhole cover", "traffic sign", "street lamp",
        "tomato plant", "tomato vine", "tomato", "soccer ball", "football", "sports ball", "ball", "car", "truck", "bus", "van",
        "motorcycle", "bicycle", "person", "child", "woman", "man", "dog", "cat", "bird",
        "horse", "cow", "sheep", "tree", "plant", "grass", "road", "fence",
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
    """按 HSV 颜色证据重排，不把颜色作为硬删除条件。"""
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
    # // 优化：颜色策略 - 所有候选都保留；HSV 覆盖率只影响展示/后续排序。
    coverage_by_index = dict(candidates)
    return sorted(indexes, key=lambda index: coverage_by_index.get(index, 0.0), reverse=True)


def referring_subject_recall(prompt: str) -> str | None:
    """从已翻译的指代短语提取主体词，供第二次同类实例召回。"""
    subjects = (
        "soccer ball", "tomato", "person", "child", "woman", "man", "dog", "cat",
        "bird", "car", "truck", "bus", "motorcycle", "bicycle", "cup", "bottle",
        "laptop", "monitor", "cell phone", "table", "chair", "apple", "banana", "book",
    )
    normalized = prompt.lower()
    for subject in subjects:
        if re.search(rf"\b{re.escape(subject)}\b", normalized):
            return subject
    return None


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


VIDEO_TEXT_LABELS = {"text", "sign text", "license plate"}
VEHICLE_LABELS = {"car", "truck", "pickup truck", "suv", "van", "bus", "motorcycle", "bicycle", "off-road vehicle"}
PERSON_LABELS = {"person", "child", "man", "woman", "pedestrian", "cyclist"}
OUTDOOR_LABELS = {"desert", "field", "sand", "mountain", "hill", "grass", "dirt road", "road", "shrub", "tree"}


def _content_objects(objects: list[DetectedObject]) -> list[DetectedObject]:
    """OCR 框不是场景主体；摘要与人数统计只用真实物体检测实例。"""
    return [item for item in objects if item.label.lower().strip() not in VIDEO_TEXT_LABELS]


def _representative_video_objects(frames: list[VideoFrameAnnotation]) -> list[DetectedObject]:
    """按单帧选择视频主体，禁止把同一人在多个抽样帧中重复累计。"""
    best: list[DetectedObject] = []
    best_rank = -1.0
    for frame in frames:
        entities = _content_objects(frame.objects)
        labels = [item.label.lower().strip() for item in entities]
        vehicles = sum(label in VEHICLE_LABELS for label in labels)
        people = sum(label in PERSON_LABELS for label in labels)
        context = sum(label in OUTDOOR_LABELS for label in labels)
        # 只要某帧检测到车辆，车辆数量就是首要排序键；避免其他帧重复出现
        # 的行人数量压过“多车荒野道路”这类实际主体场景。
        rank = vehicles * 100.0 + context * 2.0 + min(people, 3) * 0.1
        if rank > best_rank:
            best, best_rank = entities, rank
    return best


def link_video_tracks(frames: list[VideoFrameAnnotation]) -> None:
    """Assign lightweight IoU tracks to people/vehicles across sampled frames."""
    # // 优化：跨帧跟踪 - 不把同一车辆/人物在相邻关键帧重复计为新实例。
    tracked_labels = VEHICLE_LABELS | PERSON_LABELS
    active: dict[str, tuple[str, list[float]]] = {}
    sequence = 0
    for frame in frames:
        used: set[str] = set()
        for item in frame.objects:
            label = item.label.lower().strip()
            if label not in tracked_labels:
                continue
            match = next(
                (
                    track_id for track_id, (track_label, track_box) in active.items()
                    if track_id not in used and track_label == label
                    and box_iou(np.asarray(track_box, dtype=np.float32), np.asarray(item.bbox, dtype=np.float32)) >= 0.28
                ),
                None,
            )
            if match is None:
                sequence += 1
                match = f"{label}-{sequence}"
            item.track_id = match
            active[match] = (label, list(item.bbox))
            used.add(match)


def infer_activity(objects: list[DetectedObject], media_type: str) -> str:
    entities = _content_objects(objects)
    labels = [item.label.lower().strip() for item in entities]
    person_count = sum(label in PERSON_LABELS for label in labels)
    vehicle_count = sum(label in VEHICLE_LABELS for label in labels)
    ball_count = sum("ball" in label for label in labels)
    road_context = any(label in {"road", "dirt road", "sidewalk", "parking lot"} for label in labels)
    wilderness = any(label in {"desert", "field", "sand", "mountain", "hill", "grass"} for label in labels)
    if media_type == "video" and vehicle_count >= 2:
        setting = "荒野道路" if wilderness else "道路" if road_context else "画面场景"
        return f"视频关键帧中可见 {vehicle_count} 辆车辆，主体是车辆在{setting}上的行驶场景。"
    if ball_count and person_count >= 2:
        return "画面中检测到多人和球类目标，呈现球场活动场景。"
    if media_type == "video" and person_count:
        return f"视频关键帧中可见约 {person_count} 名人物；人数按单帧估计，不跨帧重复累计。"
    if person_count and road_context:
        return "画面中检测到人物与道路场景；单张图片无法可靠确认行走等动态行为。"
    if vehicle_count:
        return "画面中检测到车辆及其周边道路/环境。"
    if person_count:
        return "画面中检测到人物主体；静态图片不足以确认具体动作。"
    return "当前检测结果没有足够的时序或动作证据，暂不能确认具体行为。"


def describe_content(objects: list[DetectedObject], media_type: str) -> str:
    """按单帧主要实体概括内容，不把 OCR 文本或框面积误当作场景语义。"""
    entities = _content_objects(objects)
    labels = [item.label.lower().strip() for item in entities]
    vehicle_count = sum(label in VEHICLE_LABELS for label in labels)
    person_count = sum(label in PERSON_LABELS for label in labels)
    tomato_count = sum(label in {"tomato", "tomato plant"} for label in labels)
    plant_context = any(label in {"tree", "plant", "leaf", "grass", "bush", "shrub", "tomato plant"} for label in labels)
    road_context = any(label in {"road", "dirt road", "sidewalk", "parking lot"} for label in labels)
    wilderness = any(label in {"desert", "field", "sand", "mountain", "hill", "grass", "shrub"} for label in labels)
    if vehicle_count >= 2:
        if media_type == "video":
            setting = "荒野道路环境" if wilderness else "道路环境" if road_context else "场景中"
            return f"视频关键帧可见 {vehicle_count} 辆车辆，主要内容是{setting}中的车辆行驶。"
        return f"画面中可见 {vehicle_count} 辆车辆及其周边环境。"
    if tomato_count and plant_context:
        return f"画面中可见西红柿植株和 {tomato_count} 个西红柿。"
    if vehicle_count:
        return "画面中可见车辆及其周边环境。"
    if person_count:
        return f"画面中可见 {person_count} 名人物。"
    if labels:
        names = list(dict.fromkeys(item.description or item.label for item in entities))[:5]
        return "画面中可见" + "、".join(names) + "。"
    return "当前画面未检测到足够明确的物体主体。"


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
    """Soft-NMS: decay dense same-class overlaps rather than hard-deleting them."""
    # // 优化：自适应 NMS - 小目标阈值更低且使用分数衰减，避免密集番茄被硬删除。
    height = max(1, int(np.max(boxes[:, 3])) if len(boxes) else 1)
    width = max(1, int(np.max(boxes[:, 2])) if len(boxes) else 1)
    candidates = [Candidate(tuple(float(value) for value in boxes[index]), float(scores[index]), labels[index]) for index in range(len(boxes))]
    selected = adaptive_soft_nms(candidates, (height, width), min_score=0.03, max_count=max_count)
    # Map by a stable (box, label) key; candidate values are copied by Soft-NMS.
    result: list[int] = []
    consumed: set[int] = set()
    for item in selected:
        matches = [
            index for index in range(len(boxes))
            if index not in consumed and labels[index] == item.label and np.allclose(boxes[index], np.asarray(item.bbox), atol=0.01)
        ]
        if matches:
            result.append(matches[0])
            consumed.add(matches[0])
    return result


def keep_validated_high_risk_indexes(
    image_rgb: np.ndarray, boxes: np.ndarray, scores: np.ndarray, labels: list[str], indexes: list[int],
) -> list[int]:
    """Map category-validated candidates back to source indexes for the CPU path."""
    candidates = [Candidate(tuple(float(value) for value in boxes[index]), float(scores[index]), labels[index]) for index in indexes]
    accepted = validate_high_risk_candidates(image_rgb, candidates)
    kept: list[int] = []
    remaining = list(indexes)
    for candidate in accepted:
        match = next((index for index in remaining if labels[index] == candidate.label and np.allclose(boxes[index], np.asarray(candidate.bbox), atol=0.01)), None)
        if match is not None:
            kept.append(match)
            remaining.remove(match)
    return kept


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


def suppress_contextual_fruit_errors(boxes: np.ndarray, labels: list[str], indexes: list[int]) -> list[int]:
    """同一番茄区域被错叫成足球/head/其它果实：仅按空间重叠抑制错类。"""
    tomatoes = [index for index in indexes if labels[index].lower().strip() in {"tomato", "tomato plant"}]
    if not tomatoes:
        return indexes
    ambiguous_fruit = {"apple", "orange", "banana", "lemon", "watermelon", "grape", "strawberry", "sports ball", "soccer ball", "football", "ball", "head"}
    kept: list[int] = []
    for index in indexes:
        label = labels[index].lower().strip()
        if label in ambiguous_fruit and any(box_iou(boxes[index], boxes[tomato]) >= 0.10 for tomato in tomatoes if tomato != index):
            continue
        kept.append(index)
    return kept


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
    # // 优化：检测/分割解耦 - 分割失败时保留空 mask 槽位，不丢弃已确认检测框。
    if expected_count and masks.size == 0:
        return np.empty((expected_count, 0, 0), dtype=np.uint8)
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


def decode_video_sample_frames(upload: UploadFile, suffix: str) -> list[tuple[float, np.ndarray, bool]]:
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
            fixed_indexes = {max(0, min(frame_count - 1, round(timestamp * fps))) for timestamp in target_times}
            # // 优化：场景变化选帧 - 用缩小灰度直方图筛选突变帧，不在扫描阶段跑模型。
            scene_scores: list[tuple[float, int]] = []
            previous_hist: np.ndarray | None = None
            scan_step = max(1, round(max(1.0, fps) * VIDEO_SCENE_SCAN_INTERVAL_SECONDS))
            for index in range(0, frame_count, scan_step):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, frame_bgr = capture.read()
                if not success or frame_bgr is None:
                    continue
                gray = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
                hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
                hist = cv2.normalize(hist, None).reshape(-1)
                if previous_hist is not None:
                    score = float(cv2.compareHist(previous_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))
                    if score >= VIDEO_SCENE_CHANGE_THRESHOLD:
                        scene_scores.append((score, index))
                previous_hist = hist
            available = max(0, VIDEO_MAX_SAMPLE_COUNT - len(fixed_indexes))
            change_indexes = [index for _, index in sorted(scene_scores, reverse=True)[:available]]
            indexes = sorted(fixed_indexes | set(change_indexes))[:VIDEO_MAX_SAMPLE_COUNT]
            # 场景突变、首尾帧是重推理关键帧；基础采样帧仅做轻量检测以维持轨迹连续性。
            key_indexes = set(change_indexes)
            if indexes:
                key_indexes.update({indexes[0], indexes[-1]})
            logger.info("视频时长 %.2fs，固定帧 %d 个，场景变化补充 %d 个。", duration, len(fixed_indexes), len(change_indexes))
            samples: list[tuple[float, np.ndarray, bool]] = []
            for frame_index in indexes:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame_bgr = capture.read()
                if not success or frame_bgr is None:
                    logger.warning("跳过无法读取的视频帧：%d", frame_index)
                    continue
                timestamp = frame_index / fps if fps > 0 else float(frame_index)
                samples.append((round(timestamp, 3), cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), frame_index in key_indexes))
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
    """Recall-first post-processing shared by OpenVINO and CPU fallback."""
    if not len(boxes):
        return boxes, scores, labels
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_rgb.shape[1] - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_rgb.shape[0] - 1)
    # // 优化：跨阶段/跨切片合并 - Soft-NMS 后仅移除近乎完全重叠的父子泛类框。
    candidates = [Candidate(tuple(float(value) for value in boxes[index]), float(scores[index]), labels[index]) for index in range(len(boxes))]
    # // 优化：类别真实性校验 - 高风险开放词必须通过几何/色彩/上下文验证才进入 Soft-NMS。
    candidates = validate_high_risk_candidates(image_rgb, candidates)
    candidates = adaptive_soft_nms(candidates, image_rgb.shape[:2], min_score=0.03, max_count=MAX_OBJECTS_PER_FRAME)
    candidates = conservative_semantic_deduplicate(candidates)
    if color_hint:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        # A lightweight colour prior is deliberately bounded so it cannot turn
        # a low-confidence colour coincidence into a positive detection.
        def color_weight(candidate: Candidate) -> float:
            x1, y1, x2, y2 = [int(round(value)) for value in candidate.bbox]
            crop = hsv[max(0, y1):min(hsv.shape[0], max(y1 + 1, y2)), max(0, x1):min(hsv.shape[1], max(x1 + 1, x2))]
            if not crop.size:
                return candidate.score
            if color_hint == "red":
                evidence = (((crop[:, :, 0] <= 10) | (crop[:, :, 0] >= 170)) & (crop[:, :, 1] > 65)).mean()
            elif color_hint == "white":
                evidence = ((crop[:, :, 1] < 55) & (crop[:, :, 2] > 150)).mean()
            else:
                evidence = 0.0
            return candidate.score * (1.0 + min(0.12, float(evidence) * 0.12))
        candidates.sort(key=color_weight, reverse=True)
    else:
        candidates.sort(key=lambda item: item.score, reverse=True)
    return (
        np.asarray([item.bbox for item in candidates], dtype=np.float32),
        np.asarray([item.score for item in candidates], dtype=np.float32),
        [item.label for item in candidates],
    )


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
        self.last_diagnostics: dict[str, Any] = {"raw_candidates": [], "timings_ms": {}}

    @property
    def ready(self) -> bool:
        return self.engine.ready

    def start_background_load(self) -> None:
        self.engine.start_background_load()

    def status(self) -> dict[str, Any]:
        return self.engine.status()

    def segment_box(self, image_rgb: np.ndarray, bbox: list[float]) -> np.ndarray:
        """Route an explicit box to SAM2 without rerunning GroundingDINO."""
        resized, scale_x, scale_y = resize_for_inference(image_rgb)
        box = np.asarray([[bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]], dtype=np.float32)
        masks = self.engine.segment_boxes(resized, box)
        if not masks:
            raise RuntimeError("SAM2 未返回用户选中框的掩码")
        mask = masks[0]
        if mask.shape != image_rgb.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8)

    def analyze(
        self,
        image_rgb: np.ndarray,
        prompt: str | tuple[str, ...],
        referring_label: str | None = None,
        color_hint: str | None = None,
        run_mode: Literal["fast", "full"] = "fast",
        lightweight: bool = False,
    ) -> list[DetectedObject]:
        resized_rgb, scale_x, scale_y = resize_for_inference(image_rgb)
        engine_prompt = prompt
        if referring_label and isinstance(prompt, str):
            subject = referring_subject_recall(prompt)
            if subject:
                # 先跑属性短语，再用主体词召回同类实例；颜色后处理负责筛选。
                engine_prompt = (prompt, f"{subject}.")
        if isinstance(prompt, tuple) and prompt == UNIVERSAL_PROMPT_GROUPS:
            # 独立处理短词组，避免长类别列表超过 256 token 后截断后半部。
            engine_prompt = tuple(prompt[:MAX_UNIVERSAL_PROMPT_GROUPS])
        use_staged_recall = ENABLE_STAGED_RECALL and not lightweight and isinstance(engine_prompt, tuple)
        try:
            raw_objects = self.engine.analyze(
                resized_rgb,
                engine_prompt,
                referring_label,
                color_hint,
                staged_recall=use_staged_recall,
                full_mode=run_mode == "full",
                enable_tiles=ENABLE_TILE_RECALL,
            )
            self.last_diagnostics = dict(self.engine.last_diagnostics)
            # // 优化：可调试性 - raw_candidates 对外统一使用原图坐标。
            mapped_candidates: list[dict[str, Any]] = []
            for candidate in self.last_diagnostics.get("raw_candidates", []):
                copied = dict(candidate)
                bbox = copied.get("bbox", [])
                if isinstance(bbox, list) and len(bbox) == 4:
                    copied["bbox"] = [float(bbox[0]) / scale_x, float(bbox[1]) / scale_y, float(bbox[2]) / scale_x, float(bbox[3]) / scale_y]
                mapped_candidates.append(copied)
            self.last_diagnostics["raw_candidates"] = mapped_candidates
        except RuntimeError as exc:
            if "logits saturated" not in str(exc):
                raise
            logger.warning("OpenVINO 检测 IR 输出饱和，自动切换 PyTorch CPU 推理：%s", exc)
            raw_objects = self._fallback_analyze(resized_rgb, prompt, referring_label, color_hint)
            self.last_diagnostics = {"raw_candidates": [], "timings_ms": {"dino_detection": 0.0, "nms_postprocess": 0.0, "sam2_segmentation": 0.0}, "fallback": "pytorch"}
        # A stale FP16 IR can produce finite-but-useless logits and an empty
        # list without raising. Retry once through the known-good CPU path so
        # a successful HTTP response never masquerades as a broken analysis.
        if not raw_objects:
            logger.warning("OpenVINO 未返回目标，自动使用 PyTorch CPU 路径重试。")
            raw_objects = self._fallback_analyze(resized_rgb, prompt, referring_label, color_hint)
            self.last_diagnostics = {"raw_candidates": self.last_diagnostics.get("raw_candidates", []), "timings_ms": self.last_diagnostics.get("timings_ms", {}), "fallback": "pytorch"}
        original_height, original_width = image_rgb.shape[:2]
        restored: list[DetectedObject] = []
        posed_people = 0
        for item in raw_objects:
            source_mask = item.mask if isinstance(item, DetectedObject) else item["mask"]
            # 未进入 SAM2 分割预算的真实检测框用空字符串表示。先判断
            # 空值再转数组，避免 np.asarray("", dtype=uint8) 抛出 ValueError。
            if source_mask is None or source_mask == "":
                mask = np.empty((0, 0), dtype=np.uint8)
            else:
                mask = np.asarray(source_mask, dtype=np.uint8)
            # 低优先级目标仍保留真实检测框，但未执行 SAM2 时 mask 为空。
            # 不能把空掩码 resize 成全图伪掩码，也不能让 cv2 对一维数组报错。
            if mask.ndim != 2 or not mask.size:
                mask = np.empty((0, 0), dtype=np.uint8)
            elif mask.shape != (original_height, original_width):
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
                mask=(mask > 0).astype(np.uint8).tolist() if mask.size else "",
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
        # 仅在 CPU 兜底时裁剪通用词表。明确文本/指代提示必须保持用户原始输入。
        if isinstance(prompt, tuple) and prompt == UNIVERSAL_PROMPT_GROUPS:
            groups = prompt[:MAX_UNIVERSAL_PROMPT_GROUPS]
            prompt = tuple(groups)
            logger.info("CPU 兜底分 %d 组短提示执行检测，确保通用类别表没有被静默截断。", len(groups))
        elif referring_label and isinstance(prompt, str):
            subject = referring_subject_recall(prompt)
            if subject:
                prompt = (prompt, f"{subject}.")
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
# // 优化：异步流水线 - OCR 与检测并行，Caption 使用独立单消费者队列避免拖慢首个响应。
OCR_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, int(os.getenv("OCR_WORKERS", "1"))), thread_name_prefix="ocr-worker")
CAPTION_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="caption-worker")
caption_tasks: dict[str, dict[str, str]] = {}
caption_tasks_lock = threading.Lock()


def parse_bbox(raw_bbox: str, image: np.ndarray) -> list[float]:
    try:
        values = json.loads(raw_bbox)
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError("bbox 必须为四个数值")
        x1, y1, x2, y2 = (float(value) for value in values)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("bbox 必须为有限数值")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="bbox 必须是 JSON 数组 [x1, y1, x2, y2]。") from exc
    height, width = image.shape[:2]
    x1, x2 = max(0.0, min(x1, width - 1.0)), max(1.0, min(x2, float(width)))
    y1, y2 = max(0.0, min(y1, height - 1.0)), max(1.0, min(y2, float(height)))
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=422, detail="bbox 必须在图像范围内且右下角大于左上角。")
    return [x1, y1, x2, y2]


def resolve_ocr(future: Future[list[tuple[str, tuple[float, float, float, float]]]]) -> list[tuple[str, tuple[float, float, float, float]]]:
    try:
        return future.result()
    except Exception as exc:  # pragma: no cover - optional OCR runtime
        capability_enricher._ocr_error = str(exc)
        logger.warning("并行 OCR 失败，将返回检测结果：%s", exc)
        return []


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("启动本地视觉模型服务，模型目录：%s", MODEL_DIR)
    tracemalloc.start()
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


@app.post("/api/segment-region", response_model=SegmentResponse)
def segment_region(media: UploadFile = File(...), bbox: str = Form(...)) -> SegmentResponse:
    """Generate a SAM2 mask for one explicitly selected image box."""
    if not model_manager.ready:
        raise HTTPException(status_code=503, detail="本地模型尚未准备完成。")
    suffix = Path(media.filename or "").suffix.lower()
    if (media.content_type or "").startswith("video/") or suffix in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=415, detail="按需分割当前仅支持图片；视频请先选择关键帧导出为图片。")
    try:
        image = decode_image(media.file.read())
        requested_box = parse_bbox(bbox, image)
        started = time.perf_counter()
        mask = model_manager.segment_box(image, requested_box)
        mask_rle, mask_size = encode_mask_rle(mask)
        logger.info("用户选框 SAM2 分割完成，耗时 %.1f ms。", (time.perf_counter() - started) * 1000)
        return SegmentResponse(bbox=[round(value, 2) for value in requested_box], mask_rle=mask_rle, mask_size=mask_size)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("用户选框分割失败：%s", exc)
        raise HTTPException(status_code=500, detail=f"SAM2 分割失败：{exc}") from exc
    finally:
        media.file.close()


@app.post("/api/caption-tasks", response_model=CaptionTaskResponse)
def create_caption_task(media: UploadFile = File(...), bbox: str = Form(...)) -> CaptionTaskResponse:
    """Queue one selected object caption; never blocks detection responses."""
    if capability_enricher._caption_ir_dir() is None:
        return CaptionTaskResponse(task_id="", status="unavailable", error="Moondream2 OpenVINO Stateful IR 尚未导出，Caption 暂不可用。")
    suffix = Path(media.filename or "").suffix.lower()
    if (media.content_type or "").startswith("video/") or suffix in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=415, detail="按需 Caption 当前仅支持图片。")
    try:
        image = decode_image(media.file.read())
        x1, y1, x2, y2 = (int(value) for value in parse_bbox(bbox, image))
        crop = image[y1:y2, x1:x2].copy()
        if not crop.size:
            raise HTTPException(status_code=422, detail="选中区域为空，无法生成描述。")
        task_id = str(uuid.uuid4())
        with caption_tasks_lock:
            caption_tasks[task_id] = {"status": "queued", "caption": "", "error": ""}

        def run_caption() -> None:
            with caption_tasks_lock:
                caption_tasks[task_id]["status"] = "running"
            try:
                caption = capability_enricher.caption(crop)
                if not caption:
                    raise RuntimeError(capability_enricher._caption_error or "Caption 引擎未返回描述")
                with caption_tasks_lock:
                    caption_tasks[task_id] = {"status": "completed", "caption": caption, "error": ""}
            except Exception as exc:  # pragma: no cover - VLM optional runtime
                logger.warning("Caption 异步任务失败：%s", exc)
                with caption_tasks_lock:
                    caption_tasks[task_id] = {"status": "failed", "caption": "", "error": str(exc)}

        CAPTION_EXECUTOR.submit(run_caption)
        return CaptionTaskResponse(task_id=task_id, status="queued")
    finally:
        media.file.close()


@app.get("/api/caption-tasks/{task_id}", response_model=CaptionTaskResponse)
def get_caption_task(task_id: str) -> CaptionTaskResponse:
    with caption_tasks_lock:
        task = caption_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Caption 任务不存在或已过期。")
    return CaptionTaskResponse(task_id=task_id, status=task["status"], caption=task["caption"], error=task["error"])


@app.post("/api/analyze-media", response_model=AnalyzeResponse)
def analyze_media(
    media: UploadFile = File(...),
    text_prompt: str = Form(""),
    prompt_mode: Literal["universal", "text", "referring"] = Form("text"),
    run_mode: Literal["fast", "full"] = Form(DEFAULT_RUN_MODE),
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

    started_at = time.perf_counter()
    cpu_started = time.process_time()
    _, memory_before = tracemalloc.get_traced_memory()
    try:
        normalized_prompt, referring_label = prepare_prompt(prompt_mode, text_prompt)
        color_hint = referring_color_hint(text_prompt) if prompt_mode == "referring" else None
        media_type: Literal["image", "video"] = "video" if is_video(media, suffix) else "image"
        if media_type == "video":
            samples = decode_video_sample_frames(media, suffix or ".mp4")
            if prompt_mode == "universal":
                normalized_prompt = VIDEO_UNIVERSAL_PROMPT_GROUPS
            frames = []
            for timestamp, image_rgb, is_keyframe in samples:
                # // 优化：流水线并行 - OCR 在 GroundingDINO/SAM2 执行期间处理整图一次。
                ocr_future = OCR_EXECUTOR.submit(capability_enricher.scan_ocr, image_rgb)
                # // 优化：视频分级推理 - 场景变化关键帧走完整召回，基础帧关闭阶段补检和切片。
                detection = model_manager.analyze(
                    image_rgb, normalized_prompt, referring_label, color_hint, run_mode, lightweight=not is_keyframe,
                )
                frames.append(VideoFrameAnnotation(
                    timestamp=timestamp,
                    objects=capability_enricher.enrich(image_rgb, detection, object_factory=DetectedObject, ocr_regions=resolve_ocr(ocr_future)),
                    is_keyframe=is_keyframe,
                ))
            link_video_tracks(frames)
            objects = frames[0].objects
            all_objects = [item for frame in frames for item in frame.objects]
            summary_objects = _representative_video_objects(frames)
        else:
            preprocess_started = time.perf_counter()
            image_rgb = decode_image(media.file.read())
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
            # // 优化：流水线并行 - 不等待检测结束才启动 OCR，避免两条独立模型链路串行。
            ocr_future = OCR_EXECUTOR.submit(capability_enricher.scan_ocr, image_rgb)
            detection = model_manager.analyze(image_rgb, normalized_prompt, referring_label, color_hint, run_mode)
            objects = capability_enricher.enrich(image_rgb, detection, object_factory=DetectedObject, ocr_regions=resolve_ocr(ocr_future))
            frames = []
            all_objects = objects
            summary_objects = objects
        labels = list(dict.fromkeys(item.label for item in all_objects))
        # 视频类目可汇总所有关键帧，但主体与人数描述只能来自单个代表帧，
        # 否则同一个人会因重复抽帧被错误累计成很多人。
        activity = infer_activity(summary_objects, media_type)
        feedback = describe_content(summary_objects, media_type)
        compact_frames = [
            VideoFrameAnnotation(timestamp=frame.timestamp, objects=compact_objects(frame.objects), is_keyframe=frame.is_keyframe)
            for frame in frames
        ]
        diagnostics = getattr(model_manager, "last_diagnostics", {})
        raw_candidates = [RawCandidate(**item) for item in diagnostics.get("raw_candidates", [])[:RAW_CANDIDATE_LIMIT]]
        timings = dict(diagnostics.get("timings_ms", {}))
        timings["preprocess"] = round(preprocess_ms if media_type == "image" else 0.0, 2)
        timings["ocr"] = capability_enricher.last_metrics.get("ocr", 0.0)
        timings["caption"] = capability_enricher.last_metrics.get("caption", 0.0)
        timings["total"] = round((time.perf_counter() - started_at) * 1000, 2)
        _, memory_peak = tracemalloc.get_traced_memory()
        rss_mb, peak_rss_mb = process_memory_usage_mb()
        resources = {
            "cpu_process_ms": round((time.process_time() - cpu_started) * 1000, 2),
            "memory_peak_mb": round(memory_peak / 1024 / 1024, 2),
            "memory_delta_mb": round(max(0, memory_peak - memory_before) / 1024 / 1024, 2),
            "process_rss_mb": rss_mb,
            "process_peak_rss_mb": peak_rss_mb,
        }
        return AnalyzeResponse(
            objects=compact_objects(objects),
            labels=labels,
        feedback=feedback,
        # 前端顶部与内容摘要都优先显示可读的内容概括，实例数由语义标签和空间标注呈现。
        summary=feedback,
            task_id=str(uuid.uuid4()),
            prompt_mode=prompt_mode,
            media_type=media_type,
            frames=compact_frames,
            activity=activity,
            run_mode=run_mode,
            raw_candidates=raw_candidates,
            timings_ms=timings,
            resource_usage=resources,
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
