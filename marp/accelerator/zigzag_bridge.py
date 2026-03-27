"""Bridge between MARP's mapping representation and ZigZag's analytical model.

This module provides helpers to:

1. Run ZigZag's hardware-performance estimation on an ONNX workload
   paired with a user-supplied accelerator description.
2. Convert a :class:`~marp.mapping.core.NxModelMapping` into metadata
   that ZigZag can consume (e.g. for comparing MARP's packing-aware
   cost model against ZigZag's analytical energy/latency predictions).

ZigZag must be installed (``pip install -e third_party/zigzag``) before
importing this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx


def estimate_hardware_performance(
    workload: str | Path | onnx.ModelProto,
    accelerator: str | Path,
    mapping: str | Path,
    *,
    opt: str = "latency",
    dump_folder: str = "outputs/zigzag",
    in_memory_compute: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run ZigZag's analytical model on the given workload + accelerator.

    This is a thin wrapper around
    :pyfunc:`zigzag.api.get_hardware_performance_zigzag` that returns
    the results in a friendlier dict.

    Args:
        workload: ONNX model (path or loaded ``ModelProto``).
        accelerator: Path to a ZigZag accelerator YAML.
        mapping: Path to a ZigZag mapping YAML.
        opt: ``'energy'``, ``'latency'``, or ``'EDP'``.
        dump_folder: Where ZigZag writes its artefacts.
        in_memory_compute: Optimise for in-memory-computing architectures.
        **kwargs: Forwarded to ``get_hardware_performance_zigzag``.

    Returns:
        A dict with keys ``energy``, ``latency``, and ``cmes``
        (list of cost-model evaluations).
    """
    # Lazy import so the rest of MARP works without ZigZag installed.
    from zigzag.api import get_hardware_performance_zigzag  # type: ignore[import-untyped]

    workload_arg: str | onnx.ModelProto
    if isinstance(workload, Path):
        workload_arg = str(workload)
    else:
        workload_arg = workload

    result = get_hardware_performance_zigzag(
        workload=workload_arg,
        accelerator=str(accelerator),
        mapping=str(mapping),
        opt=opt,
        dump_folder=dump_folder,
        in_memory_compute=in_memory_compute,
        **kwargs,
    )

    # The API returns either (energy, latency, cmes) or
    # (energy, latency, extra1, extra2, cmes) depending on flags.
    if len(result) == 3:
        energy, latency, cmes = result
    else:
        energy, latency, *_, cmes = result

    return {
        "energy": energy,
        "latency": latency,
        "cmes": cmes,
    }
