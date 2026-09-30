"""Real local fallbacks for model assets that do not yet have OpenVINO IR.

The project prefers OpenVINO.  These adapters intentionally exist only for
published assets whose source format cannot be converted by the installed
OpenVINO frontend.  They never contact a network service and return an empty
result when a local model cannot be loaded.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("vision-annotator")


class LiteRTHandPoseRuntime:
    """Run the published LiteRT RTMPose-Hand model locally on CPU.

    The downloaded hand model is a genuine ``.tflite`` artifact, not ONNX.
    OpenVINO 2025 cannot convert its graph, so using LiteRT keeps the 21-point
    capability available until a compatible official ONNX checkpoint is
    supplied.  The model itself remains fully local.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._interpreter: Any | None = None
        self._input_index: int | None = None
        self._output_indexes: list[int] = []
        self._load_attempted = False
        self._lock = threading.Lock()
        self.error: str | None = None

    @property
    def asset_exists(self) -> bool:
        return self.model_path.is_file()

    @property
    def available(self) -> bool:
        return self._interpreter is not None or (self.asset_exists and not self._load_attempted)

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "engine": "litert-local",
            "openvino_ir": False,
            "model_path": str(self.model_path),
            "loaded": self._interpreter is not None,
            "error": self.error,
        }

    def _load(self) -> Any | None:
        if self._interpreter is not None:
            return self._interpreter
        if self._load_attempted:
            return None
        self._load_attempted = True
        if not self.asset_exists:
            self.error = f"缺少本地 LiteRT 手部模型：{self.model_path}"
            return None
        try:
            from ai_edge_litert.interpreter import Interpreter  # type: ignore

            interpreter = Interpreter(model_path=str(self.model_path))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            if len(input_details) != 1 or len(output_details) < 2:
                raise RuntimeError("RTMPose-Hand LiteRT 输入/输出结构不符合预期")
            self._input_index = int(input_details[0]["index"])
            self._output_indexes = [int(item["index"]) for item in output_details[:2]]
            self._interpreter = interpreter
            self.error = None
        except Exception as exc:  # Optional local dependency.
            self.error = str(exc)
            logger.warning("LiteRT 手部姿态模型未加载：%s", exc)
        return self._interpreter

    def estimate(self, image_rgb: np.ndarray, box: np.ndarray) -> list[dict[str, float]]:
        interpreter = self._load()
        if interpreter is None or self._input_index is None:
            return []
        height, width = image_rgb.shape[:2]
        x1, y1, x2, y2 = np.clip(
            np.asarray(box, dtype=np.int32),
            [0, 0, 0, 0],
            [width - 1, height - 1, width - 1, height - 1],
        )
        if x2 <= x1 or y2 <= y1:
            return []
        roi = image_rgb[y1:y2, x1:x2]
        if not roi.size:
            return []
        # The model card specifies RGB 0-255 ImageNet normalization and
        # NCHW [1, 3, 256, 256].
        resized = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        normalized = (resized - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array([58.395, 57.12, 57.375], dtype=np.float32)
        tensor = normalized.transpose(2, 0, 1)[None]
        try:
            with self._lock:
                interpreter.set_tensor(self._input_index, tensor)
                interpreter.invoke()
                simcc_x = np.asarray(interpreter.get_tensor(self._output_indexes[0]))
                simcc_y = np.asarray(interpreter.get_tensor(self._output_indexes[1]))
        except Exception as exc:
            self.error = str(exc)
            logger.warning("LiteRT 手部姿态推理失败：%s", exc)
            return []
        if simcc_x.shape[-2:] != (21, 512) or simcc_y.shape[-2:] != (21, 512):
            self.error = f"RTMPose-Hand 输出形状异常：{simcc_x.shape} / {simcc_y.shape}"
            return []
        sx, sy = simcc_x.reshape(21, 512), simcc_y.reshape(21, 512)
        xs, ys = np.argmax(sx, axis=-1) / 2.0, np.argmax(sy, axis=-1) / 2.0
        # SimCC logits are converted to a real per-axis probability peak;
        # never expose arbitrary raw logits as a confidence score.
        def peak(values: np.ndarray) -> np.ndarray:
            shifted = values - values.max(axis=-1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            return probabilities.max(axis=-1)

        confidence = np.minimum(peak(sx), peak(sy))
        return [
            {
                "x": round(float(x1 + x * (x2 - x1) / 256.0), 2),
                "y": round(float(y1 + y * (y2 - y1) / 256.0), 2),
                "score": round(float(score), 4),
            }
            for x, y, score in zip(xs, ys, confidence)
        ]


class LocalMoondreamRuntime:
    """Load the downloaded Moondream2 snapshot for genuine local CPU captioning.

    Moondream's repository ships its own Python modules and ``model.safetensors``
    rather than a pre-exported OpenVINO graph.  Loading those audited local
    modules under a private package avoids HuggingFace's dynamic-module cache
    and also forces the bundled tokenizer JSON, so no runtime download occurs.
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        # A 1.7B FP16 model needs several GB of committed RAM even before the
        # first token.  Keep this dangerous CPU fallback opt-in: an absent IR
        # must never make the whole FastAPI worker disappear under memory
        # pressure. ``caption()`` still prefers OpenVINO whenever IR exists.
        self.cpu_enabled = os.getenv("ENABLE_MOONDREAM_CPU_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}
        self._model: Any | None = None
        self._load_attempted = False
        self._lock = threading.Lock()
        self.error: str | None = None

    @property
    def weights_path(self) -> Path:
        return self.model_dir / "model.safetensors"

    @property
    def tokenizer_path(self) -> Path:
        return self.model_dir / "tokenizer.json"

    @property
    def asset_exists(self) -> bool:
        return self.weights_path.is_file() and self.tokenizer_path.is_file() and (self.model_dir / "moondream.py").is_file()

    @property
    def available(self) -> bool:
        return self._model is not None or (self.cpu_enabled and self.asset_exists and not self._load_attempted)

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "engine": "pytorch-cpu",
            "openvino_ir": False,
            "model_dir": str(self.model_dir),
            "loaded": self._model is not None,
            "cpu_fallback_enabled": self.cpu_enabled,
            "error": self.error or (None if self.cpu_enabled else "Moondream2 OpenVINO IR 尚未导出；为防止 1.7B CPU 模型耗尽内存，CPU 兜底默认关闭"),
        }

    def _import_local_module(self) -> tuple[Any, Any]:
        """Import snapshot code as an isolated package so relative imports work."""
        package_name = "_vision_annotator_moondream2"
        package = sys.modules.get(package_name)
        if package is None:
            package = ModuleType(package_name)
            package.__path__ = [str(self.model_dir)]  # type: ignore[attr-defined]
            package.__file__ = str(self.model_dir / "__init__.py")
            sys.modules[package_name] = package
        model_module = importlib.import_module(f"{package_name}.moondream")
        hf_module = importlib.import_module(f"{package_name}.hf_moondream")
        return model_module, hf_module

    def _load(self) -> Any | None:
        if self._model is not None:
            return self._model
        if self._load_attempted:
            return None
        self._load_attempted = True
        if not self.cpu_enabled:
            self.error = "Moondream2 CPU 兜底未启用"
            return None
        if not self.asset_exists:
            self.error = f"Moondream2 本地快照不完整：{self.model_dir}"
            return None
        try:
            import torch
            from tokenizers import Tokenizer

            model_module, hf_module = self._import_local_module()
            # Do not call Tokenizer.from_pretrained(): it would contact the
            # public Hub.  Replace only the snapshot module's constructor with
            # the tokenizer that was downloaded alongside this model.
            local_tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            model_module.Tokenizer = type(
                "LocalTokenizerFactory",
                (),
                {"from_pretrained": staticmethod(lambda *_args, **_kwargs: local_tokenizer)},
            )
            # This ModelScope snapshot stores the HuggingFace wrapper's
            # ``model.*`` state dict. The adjacent weights.py supports a
            # different upstream tensor naming scheme, so loading through it
            # gives a KeyError despite a healthy safetensors file. Use the
            # matching local wrapper and require an exact state-dict match.
            model = hf_module.HfMoondream(hf_module.HfConfig())
            # Do not use safetensors.torch.load_file here: that creates a
            # second ~3.8 GB state dict beside the instantiated 1.7B model
            # and causes Windows to terminate the process under common 8 GB
            # memory limits. Copy tensors one-by-one from the memory-mapped
            # safetensors file instead.
            from safetensors import safe_open

            expected = model.state_dict()
            with safe_open(str(self.weights_path), framework="pt", device="cpu") as archive:
                keys = set(archive.keys())
                expected_keys = set(expected)
                missing, unexpected = expected_keys - keys, keys - expected_keys
                if missing or unexpected:
                    raise RuntimeError(
                        "Moondream2 state dict 与快照代码不匹配："
                        f"missing={sorted(missing)[:3]} unexpected={sorted(unexpected)[:3]}"
                    )
                for name, destination in expected.items():
                    value = archive.get_tensor(name)
                    if tuple(value.shape) != tuple(destination.shape):
                        raise RuntimeError(
                            f"Moondream2 张量形状不匹配 {name}: "
                            f"{tuple(value.shape)} != {tuple(destination.shape)}"
                        )
                    destination.copy_(value.to(dtype=destination.dtype))
                    del value
            # Keep the published half-precision representation to stay within
            # commodity-memory limits. All inference remains local CPU.
            model.to("cpu").eval()
            torch.set_grad_enabled(False)
            self._model = model
            self.error = None
            logger.info("Moondream2 已从本地 safetensors 懒加载到 CPU。")
        except Exception as exc:  # Optional model/runtime dependency.
            self.error = str(exc)
            logger.warning("本地 Moondream2 未加载：%s", exc)
        return self._model

    def caption(self, image_rgb: np.ndarray) -> str:
        model = self._load()
        if model is None or not image_rgb.size:
            return ""
        try:
            from PIL import Image

            with self._lock:
                result = model.caption(Image.fromarray(image_rgb), length="short")
            return str(result.get("caption", "")).strip() if isinstance(result, dict) else ""
        except Exception as exc:
            self.error = str(exc)
            logger.warning("本地 Moondream2 Caption 推理失败：%s", exc)
            return ""
