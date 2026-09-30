"""Load the audited local Moondream2 snapshot without any Hub access."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class MoondreamLoadError(RuntimeError):
    """The local snapshot cannot be used for a faithful export."""


def _package(model_dir: Path) -> tuple[Any, Any]:
    """Import the model repository as a private package for relative imports."""
    name = "_moondream2_openvino_source"
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(model_dir)]  # type: ignore[attr-defined]
        package.__file__ = str(model_dir / "__init__.py")
        sys.modules[name] = package
    return (
        importlib.import_module(f"{name}.moondream"),
        importlib.import_module(f"{name}.hf_moondream"),
    )


def load_native_model(model_dir: Path) -> Any:
    """Build HfMoondream and copy tensors one at a time from safetensors.

    ``safetensors.torch.load_file`` temporarily allocates another full state
    dict.  A 1.7B FP16 model then needs far more than 10 GB system RAM before
    conversion begins.  This sequential copy is intentional and keeps the
    source process local and bounded by the model plus exporter workspace.
    """
    model_dir = model_dir.resolve()
    weights = model_dir / "model.safetensors"
    tokenizer = model_dir / "tokenizer.json"
    if not weights.is_file() or not tokenizer.is_file():
        raise MoondreamLoadError(f"Moondream2 snapshot is incomplete: {model_dir}")
    try:
        import torch
        from safetensors import safe_open
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - installation error
        raise MoondreamLoadError(f"Missing export dependency: {exc}") from exc

    model_module, hf_module = _package(model_dir)
    local_tokenizer = Tokenizer.from_file(str(tokenizer))
    # The upstream constructor calls Tokenizer.from_pretrained. Replace only
    # its module binding so the export never contacts HuggingFace at runtime.
    model_module.Tokenizer = type(
        "LocalTokenizerFactory", (),
        {"from_pretrained": staticmethod(lambda *_args, **_kwargs: local_tokenizer)},
    )
    try:
        model = hf_module.HfMoondream(hf_module.HfConfig())
        expected = model.state_dict()
        with safe_open(str(weights), framework="pt", device="cpu") as archive:
            keys, expected_keys = set(archive.keys()), set(expected)
            if keys != expected_keys:
                missing, unexpected = expected_keys - keys, keys - expected_keys
                raise MoondreamLoadError(
                    "Snapshot/source state mismatch; "
                    f"missing={sorted(missing)[:3]}, unexpected={sorted(unexpected)[:3]}",
                )
            for name, destination in expected.items():
                value = archive.get_tensor(name)
                if tuple(value.shape) != tuple(destination.shape):
                    raise MoondreamLoadError(f"Tensor shape mismatch for {name}")
                destination.copy_(value.to(dtype=destination.dtype))
                del value
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
    except MoondreamLoadError:
        raise
    except Exception as exc:
        raise MoondreamLoadError(f"Failed to load local Moondream2 weights: {exc}") from exc


def tokenizer_path(model_dir: Path) -> Path:
    path = model_dir.resolve() / "tokenizer.json"
    if not path.is_file():
        raise MoondreamLoadError(f"Missing tokenizer.json: {path}")
    return path
