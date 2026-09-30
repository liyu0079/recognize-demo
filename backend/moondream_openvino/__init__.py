"""Dedicated local OpenVINO exporter for the Moondream2 snapshot."""

from .kv_cache_adapter import StatefulMoondreamPipeline

__all__ = ["StatefulMoondreamPipeline"]
