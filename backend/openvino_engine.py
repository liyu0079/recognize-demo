"""OpenVINO 三模型推理引擎。

IR 文件由 export_all_models.py 生成。这个模块不包含专有 GPU 运行时调用；GPU.0 不可用或
某一模型编译/推理失败时，会只将该模型切换到 OpenVINO CPU 插件。
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

logger = logging.getLogger("vision-annotator")


class DegradedDetectionOutput(RuntimeError):
    """Grounding DINO FP16 logits saturation that would otherwise look like no detections."""


class OpenVINOModelManager:
    """按模型懒加载的 IR 管理器，避免 Arc 显存在低频请求间被长期占用。"""

    def __init__(
        self,
        model_dir: Path,
        grounding_dir: Path,
        max_idle_seconds: int,
        box_threshold: float,
        text_threshold: float,
        canonicalize_label: Callable[[str], str],
        post_filter: Callable[[np.ndarray, np.ndarray, list[str], np.ndarray, str | None], tuple[np.ndarray, np.ndarray, list[str]]],
    ) -> None:
        self.model_dir = model_dir / "openvino"
        self.grounding_dir = grounding_dir
        self.max_idle_seconds = max_idle_seconds
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.canonicalize_label = canonicalize_label
        self.post_filter = post_filter
        self.core: Any | None = None
        self.processor: Any | None = None
        self.models: dict[str, Any] = {}
        self.model_devices: dict[str, str] = {}
        self.last_used: dict[str, float] = {}
        self.load_error: str | None = None
        self.loading = False
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        # 检测与分割是接口最小可用集；人体/手部姿态均为可选增强，缺失时
        # 不阻塞基础检测分割服务启动。
        return self.processor is not None and self._ir_exists("grounding_dino") and self._ir_exists("sam2_encoder") and self._ir_exists("sam2_prompt") and self._ir_exists("sam2_decoder")

    def _ir_exists(self, name: str) -> bool:
        return (self.model_dir / f"{name}.xml").exists() and (self.model_dir / f"{name}.bin").exists()

    def _device(self) -> str:
        if self.core is not None:
            devices = self.core.available_devices
            if "GPU.0" in devices:
                return "GPU.0"
            if "GPU" in devices:
                # One Arc card is reported as GPU by some OpenVINO releases.
                return "GPU"
        return "CPU"

    def load(self) -> None:
        """只初始化 Core 和文本处理器，不编译大模型。"""
        with self._lock:
            if self.loading or self.processor is not None:
                return
            self.loading = True
        try:
            from openvino import Core

            self.core = Core()
            self.processor = AutoProcessor.from_pretrained(self.grounding_dir, local_files_only=True)
            missing = [name for name in ("grounding_dino", "sam2_encoder", "sam2_prompt", "sam2_decoder") if not self._ir_exists(name)]
            if missing:
                raise RuntimeError("缺少 OpenVINO IR：" + ", ".join(missing) + "；请先运行 export_all_models.py")
            logger.info("OpenVINO 已就绪，首选设备：%s；模型将在首次推理时加载。", self._device())
            self.load_error = None
        except Exception as exc:
            self.load_error = str(exc)
            self.core = None
            self.processor = None
            logger.exception("OpenVINO 初始化失败：%s", exc)
        finally:
            self.loading = False

    def start_background_load(self) -> None:
        if self.ready or self.loading:
            return
        threading.Thread(target=self.load, name="openvino-loader", daemon=True).start()

    def status(self) -> dict[str, Any]:
        status = "ready" if self.ready else "loading" if self.loading else "model_error" if self.load_error else "not_loaded"
        loaded = set(self.models)
        return {
            "status": status,
            "device": self._device() if self.core else "CPU",
            "engine": "openvino",
            "grounding_dino_loaded": "grounding_dino" in loaded,
            "sam2_loaded": {"sam2_encoder", "sam2_prompt", "sam2_decoder"}.issubset(loaded),
            "rtmpose_loaded": "rtmpose_tiny" in loaded,
            "rtmpose_hand_loaded": "rtmpose_hand" in loaded,
            "model_dir": str(self.model_dir),
            "model_error": self.load_error,
        }

    def _compiled(self, name: str) -> Any:
        with self._lock:
            self.release_idle_models()
            if name in self.models:
                self.last_used[name] = time.monotonic()
                return self.models[name]
            if self.core is None:
                raise RuntimeError("OpenVINO Core 尚未初始化")
            model_path = self.model_dir / f"{name}.xml"
            preferred = self._device()
            try:
                compiled = self.core.compile_model(str(model_path), preferred)
                device = preferred
            except Exception as exc:
                if preferred == "CPU":
                    raise RuntimeError(f"{name} CPU 加载失败：{exc}") from exc
                logger.warning("%s 无法加载到 GPU.0，将回退 CPU：%s", name, exc)
                compiled = self.core.compile_model(str(model_path), "CPU")
                device = "CPU"
            self.models[name] = compiled
            self.model_devices[name] = device
            self.last_used[name] = time.monotonic()
            logger.info("OpenVINO %s 已加载到 %s。", name, device)
            return compiled

    def _run(self, name: str, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """GPU 推理阶段失败时，将单个 IR 重建到 CPU 后重试一次。"""
        compiled = self._compiled(name)
        try:
            return self._outputs(compiled(self._input_map(compiled, values)))
        except Exception as exc:
            if not self.model_devices.get(name, "").startswith("GPU") or self.core is None:
                raise
            logger.warning("%s 在 GPU.0 推理失败，回退 CPU 重试：%s", name, exc)
            with self._lock:
                self.models.pop(name, None)
                cpu_compiled = self.core.compile_model(str(self.model_dir / f"{name}.xml"), "CPU")
                self.models[name] = cpu_compiled
                self.model_devices[name] = "CPU"
                self.last_used[name] = time.monotonic()
            return self._outputs(cpu_compiled(self._input_map(cpu_compiled, values)))

    def release_idle_models(self, force: bool = False) -> None:
        """释放已闲置模型的编译图和显存；下一次请求会透明重载。"""
        now = time.monotonic()
        expired = [name for name, used in self.last_used.items() if force or now - used >= self.max_idle_seconds]
        for name in expired:
            self.models.pop(name, None)
            self.model_devices.pop(name, None)
            self.last_used.pop(name, None)
        if expired:
            gc.collect()
            logger.info("已释放闲置 OpenVINO 模型：%s", ", ".join(expired))

    @staticmethod
    def _input_map(compiled: Any, values: dict[str, np.ndarray]) -> dict[Any, np.ndarray]:
        result: dict[Any, np.ndarray] = {}
        used: set[str] = set()
        for port in compiled.inputs:
            name = port.get_any_name()
            if name not in values:
                # ONNX may rename input_ids to a numeric token id. Match that
                # port explicitly; other single-input graphs use the only value.
                if name.isdigit() and "input_ids" in values:
                    result[port] = values["input_ids"]
                    used.add("input_ids")
                elif len(values) == 1:
                    key, value = next(iter(values.items()))
                    result[port] = value
                    used.add(key)
                else:
                    raise RuntimeError(f"IR 输入 {name} 缺失；请使用本项目 export_all_models.py 重新导出。")
            else:
                result[port] = values[name]
                used.add(name)
        return result

    @staticmethod
    def _outputs(result: Any) -> dict[str, np.ndarray]:
        outputs: dict[str, np.ndarray] = {}
        for port, value in result.items():
            array = np.asarray(value)
            # 某些 OpenVINO 版本的 compiled result 会把内部数字名作为
            # get_any_name()，但 get_names() 仍保留导出时的业务别名。
            names = set(port.get_names()) or {port.get_any_name()}
            names.add(port.get_any_name())
            for name in names:
                outputs[name] = array
        return outputs

    @staticmethod
    def _find_output(outputs: dict[str, np.ndarray], *names: str) -> np.ndarray:
        for name in names:
            if name in outputs:
                return outputs[name]
        for key, value in outputs.items():
            if any(name in key for name in names):
                return value
        raise RuntimeError(f"IR 输出不包含 {names}，实际输出：{list(outputs)}")

    def _detect(self, image: np.ndarray, prompt: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        assert self.processor is not None
        # DINO IR 固定为 1200x800 / 256 tokens；检测框在后处理阶段映射回工作图。
        fixed_image = cv2.resize(image, (1200, 800), interpolation=cv2.INTER_LINEAR)
        inputs = self.processor(
            images=Image.fromarray(fixed_image),
            text=prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        values = {key: value.detach().cpu().numpy() for key, value in inputs.items()}
        outputs = self._run("grounding_dino", values)
        logits = self._find_output(outputs, "logits")
        # FP16-compressed Grounding DINO IR can saturate almost every logit at
        # -65504 (or -inf on CPU). post_process then returns an empty list and
        # makes a healthy request look like a valid "no object" result.
        saturated = np.count_nonzero(~np.isfinite(logits) | (logits <= -65000))
        if saturated / logits.size > 0.9:
            raise DegradedDetectionOutput("Grounding DINO OpenVINO logits saturated")
        result = SimpleNamespace(
            logits=torch.from_numpy(logits),
            pred_boxes=torch.from_numpy(self._find_output(outputs, "pred_boxes", "boxes")),
        )
        post_process = self.processor.post_process_grounded_object_detection
        kwargs = {"box_threshold": self.box_threshold}
        try:
            processed = post_process(result, inputs["input_ids"], self.text_threshold, [(image.shape[0], image.shape[1])], **kwargs)[0]
        except TypeError:
            processed = post_process(result, inputs["input_ids"], threshold=self.box_threshold, text_threshold=self.text_threshold, target_sizes=[(image.shape[0], image.shape[1])])[0]
        boxes = processed["boxes"].float().cpu().numpy()
        scores = processed["scores"].float().cpu().numpy()
        raw_labels = processed.get("text_labels", processed.get("labels", []))
        return boxes, scores, [self.canonicalize_label(str(label)) for label in raw_labels]

    @staticmethod
    def _sam_image(image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        normalized = (resized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return normalized.transpose(2, 0, 1)[None]

    def _segment(self, image: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
        """SAM2 三段 IR：图片编码一次，每个 bbox 用固定单框 decoder，消除动态流。"""
        image_out = self._run("sam2_encoder", {"image": self._sam_image(image)})
        image_embed = self._find_output(image_out, "image_embed")
        high_0 = self._find_output(image_out, "high_res_0")
        high_1 = self._find_output(image_out, "high_res_1")
        height, width = image.shape[:2]
        masks: list[np.ndarray] = []
        for box in boxes:
            scaled = np.array([[box[0] / width * 1024, box[1] / height * 1024, box[2] / width * 1024, box[3] / height * 1024]], dtype=np.float32)
            prompt_out = self._run("sam2_prompt", {"boxes": scaled})
            decoder_values = {
                "image_embed": image_embed,
                "high_res_0": high_0,
                "high_res_1": high_1,
                "sparse_embeddings": self._find_output(prompt_out, "sparse_embeddings"),
                "dense_embeddings": self._find_output(prompt_out, "dense_embeddings"),
            }
            decoded = self._run("sam2_decoder", decoder_values)
            logits = self._find_output(decoded, "low_res_masks", "masks")
            mask = cv2.resize(np.squeeze(logits)[-1] if np.squeeze(logits).ndim == 3 else np.squeeze(logits), (width, height), interpolation=cv2.INTER_LINEAR) > 0.0
            masks.append(mask.astype(np.uint8))
        return masks

    def _pose(self, image: np.ndarray, box: np.ndarray) -> list[dict[str, float]]:
        """RTMPose-tiny SimCC 解码；关键点从 ROI 映射回缩放后的完整图坐标。"""
        x1, y1, x2, y2 = np.clip(box.astype(np.int32), [0, 0, 0, 0], [image.shape[1] - 1, image.shape[0] - 1, image.shape[1] - 1, image.shape[0] - 1])
        if x2 <= x1 or y2 <= y1:
            return []
        roi = image[y1:y2, x1:x2]
        input_w, input_h = 192, 256
        resized = cv2.resize(roi, (input_w, input_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        normalized = ((resized / 255.0) - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        outputs = self._run("rtmpose_tiny", {"input": normalized.transpose(2, 0, 1)[None]})
        simcc_x = self._find_output(outputs, "simcc_x")
        simcc_y = self._find_output(outputs, "simcc_y")
        xs = np.argmax(simcc_x, axis=-1).reshape(-1) / 2.0
        ys = np.argmax(simcc_y, axis=-1).reshape(-1) / 2.0
        scores = np.minimum(np.max(simcc_x, axis=-1).reshape(-1), np.max(simcc_y, axis=-1).reshape(-1))
        return [{"x": round(float(x1 + x * (x2 - x1) / input_w), 2), "y": round(float(y1 + y * (y2 - y1) / input_h), 2), "score": round(float(score), 4)} for x, y, score in zip(xs[:17], ys[:17], scores[:17])]

    def estimate_pose(self, image: np.ndarray, box: np.ndarray) -> list[dict[str, float]]:
        """Public pose entry point used to enrich PyTorch fallback detections."""
        if not self.ready:
            return []
        with self._inference_lock:
            return self._pose(image, box)

    def estimate_hand_pose(self, image: np.ndarray, box: np.ndarray) -> list[dict[str, float]]:
        """Decode an optional RTMPose-hand SimCC IR; absent IR returns no points."""
        if not self.ready or not self._ir_exists("rtmpose_hand"):
            return []
        x1, y1, x2, y2 = np.clip(box.astype(np.int32), [0, 0, 0, 0], [image.shape[1] - 1, image.shape[0] - 1, image.shape[1] - 1, image.shape[0] - 1])
        if x2 <= x1 or y2 <= y1:
            return []
        roi = image[y1:y2, x1:x2]
        resized = cv2.resize(roi, (192, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        normalized = ((resized / 255.0) - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        outputs = self._run("rtmpose_hand", {"input": normalized.transpose(2, 0, 1)[None]})
        simcc_x = self._find_output(outputs, "simcc_x")
        simcc_y = self._find_output(outputs, "simcc_y")
        xs = np.argmax(simcc_x, axis=-1).reshape(-1) / 2.0
        ys = np.argmax(simcc_y, axis=-1).reshape(-1) / 2.0
        scores = np.minimum(np.max(simcc_x, axis=-1).reshape(-1), np.max(simcc_y, axis=-1).reshape(-1))
        return [{"x": round(float(x1 + x * (x2 - x1) / 192), 2), "y": round(float(y1 + y * (y2 - y1) / 256), 2), "score": round(float(score), 4)} for x, y, score in zip(xs[:21], ys[:21], scores[:21])]

    def analyze(self, image: np.ndarray, prompts: str | tuple[str, ...], referring_label: str | None = None, color_hint: str | None = None) -> list[dict[str, Any]]:
        if not self.ready:
            raise RuntimeError(self.load_error or "OpenVINO 模型尚未准备完成")
        with self._inference_lock:
            all_boxes: list[np.ndarray] = []
            all_scores: list[np.ndarray] = []
            labels: list[str] = []
            for prompt in (prompts,) if isinstance(prompts, str) else prompts:
                boxes, scores, batch_labels = self._detect(image, prompt)
                all_boxes.append(boxes)
                all_scores.append(scores)
                labels.extend(batch_labels)
            boxes = np.concatenate(all_boxes) if all_boxes else np.empty((0, 4), dtype=np.float32)
            scores = np.concatenate(all_scores) if all_scores else np.empty((0,), dtype=np.float32)
            if not len(boxes):
                return []
            boxes, scores, labels = self.post_filter(boxes, scores, labels, image, color_hint)
            masks = self._segment(image, boxes)
            objects: list[dict[str, Any]] = []
            for box, score, label, mask in zip(boxes, scores, labels, masks):
                final_label = referring_label or label
                try:
                    keypoints = self._pose(image, box) if final_label.lower() == "person" and self._ir_exists("rtmpose_tiny") else []
                except Exception as exc:
                    logger.warning("RTMPose-tiny 不可用，将跳过人体姿态：%s", exc)
                    keypoints = []
                hand_keypoints = []
                if final_label.lower() in {"hand", "hands"} and self._ir_exists("rtmpose_hand"):
                    points = self.estimate_hand_pose(image, box)
                    hand_keypoints = [points] if len(points) == 21 else []
                objects.append({"label": final_label, "score": round(float(score), 4), "bbox": [round(float(v), 2) for v in box], "mask": mask.tolist(), "keypoints": keypoints, "hand_keypoints": hand_keypoints})
            return objects
