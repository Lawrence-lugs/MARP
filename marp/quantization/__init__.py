"""Quantization utilities — quantised tensors, fixed-point conversion, and packing."""

from __future__ import annotations

from marp.quantization.quant import QuantizedTensor, convert_scale_to_shift_and_m0

__all__ = [
    "QuantizedTensor",
    "convert_scale_to_shift_and_m0",
]
