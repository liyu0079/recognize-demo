"""Moondream2 SigLIP vision and projection OpenVINO exporter.

Inputs are BGR uint8 NCHW.  The graph performs BGR-to-RGB, bilinear resize,
0..1 scaling and the Moondream2 ``(x - .5) / .5`` normalization itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class VisionExportError(RuntimeError):
    pass


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left, right = left.astype(np.float64).ravel(), right.astype(np.float64).ravel()
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))


def _wrappers(native: Any) -> tuple[Any, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from _moondream2_openvino_source.vision import vision_encoder, vision_projection

    class VisionEncoder(nn.Module):
        """The complete SigLIP backbone including IR-native preprocessing ops."""
        def __init__(self) -> None:
            super().__init__()
            self.vision = native.model.vision
            self.config = native.model.config.vision

        def forward(self, bgr_u8: torch.Tensor) -> torch.Tensor:
            # Exporting these standard Torch operations makes them OpenVINO
            # Convert/Interpolate/Multiply/Add nodes in the resulting IR.
            rgb = bgr_u8[:, [2, 1, 0], :, :].to(dtype=torch.float32)
            rgb = F.interpolate(rgb, size=(378, 378), mode="bilinear", align_corners=False)
            normalized = (rgb / 255.0 - 0.5) / 0.5
            return vision_encoder(normalized.to(dtype=self.vision.pos_emb.dtype), self.vision, self.config)

    class VisionProjector(nn.Module):
        """Exact projector for a global crop plus one 27x27 local crop grid.

        Moondream's standard 378x378 path has a 1x1 tiling. Larger images are
        tiled by the runtime, encoded as a batch, then supplied as reconstructed
        local features to this graph. No visual model weights are duplicated.
        """
        def __init__(self) -> None:
            super().__init__()
            self.vision = native.model.vision
            self.config = native.model.config.vision

        def forward(self, global_features: torch.Tensor, local_features: torch.Tensor) -> torch.Tensor:
            # Encoder batches have shape [B, 729, 1152]. The native projector
            # receives the global crop as [729, 1152] and reconstructed local
            # features as [27, 27, 1152]. The 378 path has exactly one local
            # tile, so reshaping is lossless.
            global_grid = global_features[0]
            local_grid = local_features[0].reshape(27, 27, self.config.enc_dim)
            return vision_projection(global_grid, local_grid, self.vision, self.config).unsqueeze(0)

    return VisionEncoder(), VisionProjector()


def export_vision(native: Any, output_dir: Path, validation_threshold: float = 0.999) -> dict[str, Path]:
    """Export FP16 vision IRs and compare an OpenVINO result with PyTorch."""
    import openvino as ov
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    encoder, projector = _wrappers(native)
    encoder.eval()
    projector.eval()
    sample_image = torch.randint(0, 256, (1, 3, 378, 378), dtype=torch.uint8)
    with torch.inference_mode():
        reference_encoder = encoder(sample_image).float().cpu().numpy()
        # 378 uses one local crop; same feature tensor is the exact 27x27
        # reconstruction for the native 1x1-tile code path.
        reference_prefix = projector(
            torch.from_numpy(reference_encoder), torch.from_numpy(reference_encoder),
        ).float().cpu().numpy()

    try:
        encoder_ov = ov.convert_model(encoder, example_input=sample_image)
        projector_ov = ov.convert_model(
            projector,
            example_input=(
                torch.from_numpy(reference_encoder).to(dtype=native.model.vision.pos_emb.dtype),
                torch.from_numpy(reference_encoder).to(dtype=native.model.vision.pos_emb.dtype),
            ),
        )
        # Keep crop batch dynamic while fixing model patch dimensions.
        encoder_ov.reshape({encoder_ov.input(0): ov.PartialShape([-1, 3, -1, -1])})
        ov.save_model(encoder_ov, output_dir / "vision_encoder.xml", compress_to_fp16=True)
        ov.save_model(projector_ov, output_dir / "vision_projector.xml", compress_to_fp16=True)
    except Exception as exc:
        raise VisionExportError(f"Vision conversion failed: {exc}") from exc

    compiled_encoder = ov.Core().compile_model(str(output_dir / "vision_encoder.xml"), "CPU")
    actual_encoder = next(iter(compiled_encoder({compiled_encoder.input(0): sample_image.numpy()}).values()))
    encoder_similarity = _cosine(reference_encoder, actual_encoder)
    compiled_projector = ov.Core().compile_model(str(output_dir / "vision_projector.xml"), "CPU")
    actual_prefix = next(iter(compiled_projector([actual_encoder, actual_encoder]).values()))
    prefix_similarity = _cosine(reference_prefix, actual_prefix)
    if min(encoder_similarity, prefix_similarity) < validation_threshold:
        raise VisionExportError(
            f"Vision validation failed: encoder={encoder_similarity:.6f}, "
            f"projector={prefix_similarity:.6f}, threshold={validation_threshold}",
        )
    return {
        "encoder": output_dir / "vision_encoder.xml",
        "projector": output_dir / "vision_projector.xml",
    }
