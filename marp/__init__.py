"""MARP — Mapping Accelerator for Reconfigurable Packing.

Bin-packing DNN layers onto Analog In-Memory Computing (AIMC) arrays
to maximise weight reuse across layers.
"""

from __future__ import annotations

__version__ = "0.1.0"

from marp.mapping.core import MappedBin, MappedNode, NxModelMapping, QRAccModel
from marp.mapping.packer_utils import get_packer_by_type

__all__ = [
    "MappedBin",
    "MappedNode",
    "NxModelMapping",
    "QRAccModel",
    "get_packer_by_type",
]
