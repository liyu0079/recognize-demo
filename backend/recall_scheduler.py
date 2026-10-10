"""Recall-first scheduling and post-processing for the detection layer.

The module deliberately has no model/runtime dependency.  Keeping scheduling
separate from GroundingDINO and SAM2 makes it possible to regression-test the
policy with fixed candidates and to roll it back with environment switches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np


# // 优化：阶段提示词调度 - 每阶段短提示，避免把十组词一次送入 256-token 编码器。
STAGE_TWO_PROMPTS: dict[str, tuple[str, ...]] = {
    # // 优化：高风险类别调度 - 车牌、井盖、空调外机不因泛场景词被反复补检。
    "vehicle": ("vehicle rear. vehicle front. taxi. delivery truck.",),
    "road": ("bollard. curb. lane marking. traffic cone. drainage grate.",),
    "food": ("tomato. ripe tomato. cluster of tomatoes. tomato fruit. apple. orange.",),
    "plant": ("tomato. tomato plant. flower. leaf. fruit. small plant.",),
    "electronics": ("monitor. cell phone. camera. electrical cabinet.",),
}

SMALL_OBJECT_PROMPTS = (
    "traffic sign. traffic light. bollard. drainage grate. small vehicle. bicycle.",
    "tomato. ripe tomato. cluster of tomatoes. motorcycle. cell phone.",
)

VEHICLE_LABELS = {"car", "truck", "pickup truck", "suv", "van", "bus", "motorcycle", "bicycle", "vehicle"}
ROAD_LABELS = {"road", "dirt road", "sidewalk", "crosswalk", "parking lot", "traffic sign", "traffic light", "guardrail", "street lamp"}
FOOD_LABELS = {"tomato", "tomato plant", "fruit", "apple", "orange", "banana", "lemon"}
PLANT_LABELS = {"plant", "tree", "flower", "leaf", "grass", "bush", "shrub", "tomato plant"}
ELECTRONICS_LABELS = {"air conditioner", "outdoor air conditioner", "monitor", "laptop", "cell phone", "television"}


@dataclass(frozen=True)
class Candidate:
    """A traceable detection candidate before and after post-processing."""

    bbox: tuple[float, float, float, float]
    score: float
    label: str
    stage: int = 1
    tile_id: str = "full"

    def as_dict(self) -> dict[str, object]:
        return {
            "bbox": [round(float(value), 2) for value in self.bbox],
            "score": round(float(self.score), 4),
            "label": self.label,
            "stage": self.stage,
            "tile_id": self.tile_id,
        }


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    left, top, right, bottom = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def area_ratio(candidate: Candidate, image_shape: tuple[int, int]) -> float:
    height, width = image_shape
    x1, y1, x2, y2 = candidate.bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, float(height * width))


def prompt_groups_for_labels(labels: Iterable[str]) -> tuple[str, ...]:
    labels = {label.lower().strip() for label in labels}
    groups: list[str] = []
    if labels & VEHICLE_LABELS:
        groups.extend(STAGE_TWO_PROMPTS["vehicle"])
    if labels & ROAD_LABELS:
        groups.extend(STAGE_TWO_PROMPTS["road"])
    # A first-pass tomato hallucination must not unlock a second pass that
    # produces more fruit hallucinations. Food details require plant context.
    if labels & PLANT_LABELS:
        groups.extend(STAGE_TWO_PROMPTS["food"])
    if labels & PLANT_LABELS:
        groups.extend(STAGE_TWO_PROMPTS["plant"])
    if labels & ELECTRONICS_LABELS:
        groups.extend(STAGE_TWO_PROMPTS["electronics"])
    return tuple(dict.fromkeys(groups))


HIGH_RISK_LABELS = {"license plate", "manhole cover", "air conditioner", "tomato"}


def _crop(image: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    return image[max(0, y1):min(height, max(y1 + 1, y2)), max(0, x1):min(width, max(x1 + 1, x2))]


def _geometry(candidate: Candidate, image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = image_shape
    x1, y1, x2, y2 = candidate.bbox
    box_width, box_height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return box_width / box_height, area_ratio(candidate, image_shape), (x1 + x2) / 2 / width, (y1 + y2) / 2 / height


def _tomato_blob(crop: np.ndarray) -> bool:
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask = (((hsv[:, :, 0] <= 16) | (hsv[:, :, 0] >= 165) | ((hsv[:, :, 0] >= 17) & (hsv[:, :, 0] <= 38))) & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 65)).astype(np.uint8)
    if float(mask.mean()) < 0.10:
        return False
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    if not width or not height:
        return False
    ratio = width / height
    fill = cv2.contourArea(contour) / float(width * height)
    return 0.62 <= ratio <= 1.55 and fill >= 0.42


def _low_saturation_ratio(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    return float(((hsv[:, :, 1] <= 72) & (hsv[:, :, 2] >= 45)).mean())


def validate_high_risk_candidates(image: np.ndarray, candidates: list[Candidate]) -> list[Candidate]:
    """Reject only category-contradictory open-vocabulary matches.

    Raw candidates remain available in the API diagnostics. This validation is
    deliberately narrow: ordinary objects retain recall-first Soft-NMS.
    """
    vehicle_boxes = [item.bbox for item in candidates if item.label.lower().strip() in VEHICLE_LABELS]
    accepted: list[Candidate] = []
    for candidate in candidates:
        label = candidate.label.lower().strip()
        if label not in HIGH_RISK_LABELS:
            accepted.append(candidate)
            continue
        ratio, area, _, center_y = _geometry(candidate, image.shape[:2])
        crop = _crop(image, candidate.bbox)
        if label == "license plate":
            # // 优化：车牌上下文校验 - 没有车辆承载关系的绿色叶片/招牌不能是车牌。
            contained = any(
                max(0.0, min(candidate.bbox[2], box[2]) - max(candidate.bbox[0], box[0]))
                * max(0.0, min(candidate.bbox[3], box[3]) - max(candidate.bbox[1], box[1]))
                / max(1.0, (candidate.bbox[2] - candidate.bbox[0]) * (candidate.bbox[3] - candidate.bbox[1])) >= 0.72
                for box in vehicle_boxes
            )
            if contained and 1.35 <= ratio <= 7.5 and area <= 0.018:
                accepted.append(candidate)
        elif label == "manhole cover":
            # // 优化：井盖形态校验 - 人体、道路大框和墙面框不满足地面小型近方形结构。
            if candidate.score >= 0.40 and center_y >= 0.52 and 0.65 <= ratio <= 1.6 and 0.00025 <= area <= 0.025 and _low_saturation_ratio(crop) >= 0.42:
                accepted.append(candidate)
        elif label == "air conditioner":
            # // 优化：空调外机形态校验 - 排除整扇窗、整辆摩托和整片立面的大框。
            height, width = image.shape[:2]
            box_width, box_height = candidate.bbox[2] - candidate.bbox[0], candidate.bbox[3] - candidate.bbox[1]
            if candidate.score >= 0.42 and 0.50 <= ratio <= 2.35 and 0.0004 <= area <= 0.045 and box_width <= width * 0.22 and box_height <= height * 0.22 and center_y <= 0.84 and _low_saturation_ratio(crop) >= 0.20:
                accepted.append(candidate)
        elif label == "tomato":
            # // 优化：番茄颜色与轮廓校验 - 红色招牌、文字块和街景反光不会作为果实输出。
            if candidate.score >= 0.28 and 0.00008 <= area <= 0.065 and 0.62 <= ratio <= 1.55 and _tomato_blob(crop):
                accepted.append(candidate)
    return accepted


def should_run_tiles(candidates: list[Candidate], image_shape: tuple[int, int], full_mode: bool) -> bool:
    """Run only a targeted local pass when the first pass suggests it helps."""
    if full_mode:
        return True
    if not candidates:
        return True
    coverage = sum(area_ratio(item, image_shape) for item in candidates)
    small = sum(1 for item in candidates if area_ratio(item, image_shape) < 0.012)
    # Sparse coverage can hide distant objects; clusters of small boxes are a
    # separate signal for dense fruits, signs, or plates.
    return coverage < 0.18 or small >= 4


def image_tiles(image: np.ndarray, grid: int) -> list[tuple[str, np.ndarray, int, int]]:
    """Build slightly-overlapped tiles so edge objects are not split away."""
    height, width = image.shape[:2]
    overlap_x, overlap_y = max(1, width // (grid * 12)), max(1, height // (grid * 12))
    result: list[tuple[str, np.ndarray, int, int]] = []
    for row in range(grid):
        for column in range(grid):
            x1 = max(0, column * width // grid - overlap_x)
            y1 = max(0, row * height // grid - overlap_y)
            x2 = min(width, (column + 1) * width // grid + overlap_x)
            y2 = min(height, (row + 1) * height // grid + overlap_y)
            result.append((f"{row}-{column}", image[y1:y2, x1:x2], x1, y1))
    return result


def map_tile_candidate(candidate: Candidate, offset_x: int, offset_y: int, tile_id: str) -> Candidate:
    x1, y1, x2, y2 = candidate.bbox
    return Candidate((x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y), candidate.score, candidate.label, candidate.stage, tile_id)


def adaptive_soft_nms(candidates: list[Candidate], image_shape: tuple[int, int], min_score: float, max_count: int) -> list[Candidate]:
    """Class-aware Gaussian Soft-NMS with smaller-object-friendly thresholds.

    Unlike hard NMS, candidate boxes are retained and only their rank score is
    decayed.  This prevents neighbouring tomatoes and distant vehicles from
    disappearing solely because their boxes overlap.
    """
    remaining = list(candidates)
    selected: list[Candidate] = []
    while remaining and len(selected) < max_count:
        remaining.sort(key=lambda item: item.score, reverse=True)
        best = remaining.pop(0)
        if best.score < min_score:
            break
        selected.append(best)
        updated: list[Candidate] = []
        best_small = area_ratio(best, image_shape) < 0.012
        threshold = 0.32 if best_small else 0.58
        for item in remaining:
            if item.label.lower().strip() != best.label.lower().strip():
                updated.append(item)
                continue
            overlap = box_iou(best.bbox, item.bbox)
            if overlap <= threshold:
                updated.append(item)
                continue
            # A gentle decay lets dense, valid instances survive when their
            # initial confidence is adequate, while exact duplicate boxes sink.
            sigma = 0.48 if best_small else 0.32
            score = item.score * float(np.exp(-((overlap - threshold) ** 2) / sigma))
            updated.append(Candidate(item.bbox, score, item.label, item.stage, item.tile_id))
        remaining = updated
    return selected


def conservative_semantic_deduplicate(candidates: list[Candidate]) -> list[Candidate]:
    """Remove only near-identical parent/child duplicates, never low-IoU fruit boxes."""
    parent_children = {
        "vehicle": VEHICLE_LABELS - {"vehicle"},
        "fruit": FOOD_LABELS - {"fruit"},
        "plant": PLANT_LABELS - {"plant"},
    }
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        label = candidate.label.lower().strip()
        duplicate = False
        for existing in kept:
            other = existing.label.lower().strip()
            related = other in parent_children.get(label, set()) or label in parent_children.get(other, set())
            if related and box_iou(candidate.bbox, existing.bbox) >= 0.82:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


Detector = Callable[[np.ndarray, str], list[Candidate]]


def staged_recall_detect(
    image: np.ndarray,
    stage_one_prompts: tuple[str, ...],
    detector: Detector,
    *,
    full_mode: bool,
    enable_tiles: bool,
) -> tuple[list[Candidate], list[Candidate]]:
    """Execute 1) base, 2) label-driven, and 3) local tile recall passes."""
    stage_one = [item for prompt in stage_one_prompts for item in detector(image, prompt)]
    stage_two_prompts = prompt_groups_for_labels(item.label for item in stage_one)
    stage_two = [Candidate(item.bbox, item.score, item.label, 2, item.tile_id) for prompt in stage_two_prompts for item in detector(image, prompt)]
    raw = stage_one + stage_two
    if not enable_tiles or not should_run_tiles(raw, image.shape[:2], full_mode):
        return raw, raw
    grid = 3 if full_mode else 2
    tiled: list[Candidate] = []
    prompts = stage_two_prompts or SMALL_OBJECT_PROMPTS
    for tile_id, tile, x_offset, y_offset in image_tiles(image, grid):
        if tile.size == 0:
            continue
        # Upscaling only the local input preserves global latency and makes the
        # fixed 1200x800 detector see remote objects at useful pixel sizes.
        enlarged = cv2.resize(tile, None, fx=1.45, fy=1.45, interpolation=cv2.INTER_CUBIC)
        for prompt in tuple(dict.fromkeys(prompts + SMALL_OBJECT_PROMPTS)):
            for item in detector(enlarged, prompt):
                x1, y1, x2, y2 = (value / 1.45 for value in item.bbox)
                tiled.append(map_tile_candidate(Candidate((x1, y1, x2, y2), item.score, item.label, 3), x_offset, y_offset, tile_id))
    return raw + tiled, raw
