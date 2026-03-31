"""Tests for running non-split ONNX workloads on builtin ZigZag accelerator architectures.

ZigZag's ONNX parser supports QLinearConv, Conv, MatMul, and Gemm operators.
The ``ad`` model originally used QLinearMatMul; a converted version
(``ad_quantized_int8_conv.onnx``) with equivalent pointwise QLinearConv nodes
is used here so that ZigZag can fully evaluate it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper, shape_inference

from zigzag.api import get_hardware_performance_zigzag


def _inferred_model(path: str) -> onnx.ModelProto:
    """Load an ONNX model with full shape information.

    Standard ``onnx.shape_inference`` cannot resolve QLinear* operators, so we
    fall back to an ORT-based probe: run the model with dummy data to discover
    intermediate tensor shapes, then attach them as ``value_info`` entries.
    """
    import onnxruntime as ort

    model = onnx.load(path)

    # First try standard shape inference
    model = shape_inference.infer_shapes(model)

    # Collect names already known *with valid shapes* (all dims > 0)
    known = set()
    for i in model.graph.input:
        known.add(i.name)
    for o in model.graph.output:
        known.add(o.name)
    for init in model.graph.initializer:
        known.add(init.name)

    # value_info entries with any 0-dim are unreliable (shape inference
    # could not resolve them through QLinear* ops) – drop those.
    good_vi: list[onnx.ValueInfoProto] = []
    bad_vi_names: set[str] = set()
    for vi in model.graph.value_info:
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if any(d == 0 for d in dims):
            bad_vi_names.add(vi.name)
        else:
            known.add(vi.name)
            good_vi.append(vi)

    if bad_vi_names:
        del model.graph.value_info[:]
        model.graph.value_info.extend(good_vi)

    # Find unresolved intermediate tensors
    unresolved = []
    for node in model.graph.node:
        for out in node.output:
            if out and out not in known:
                unresolved.append(out)

    if not unresolved:
        return model

    # Probe with ORT to discover shapes
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    for name in unresolved:
        vi = helper.ValueInfoProto()
        vi.name = name
        probe.graph.output.append(vi)

    session = ort.InferenceSession(probe.SerializeToString())
    inp_meta = session.get_inputs()[0]
    dummy = np.random.randn(*inp_meta.shape).astype(np.float32)
    results = session.run(unresolved, {inp_meta.name: dummy})

    for name, arr in zip(unresolved, results):
        elem_type = onnx.helper.np_dtype_to_tensor_dtype(arr.dtype)
        vi = helper.make_tensor_value_info(name, elem_type, list(arr.shape))
        model.graph.value_info.append(vi)

    return model

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "onnx_models"
ZIGZAG_DIR = REPO_ROOT / "third_party" / "zigzag"
HW_DIR = ZIGZAG_DIR / "zigzag" / "inputs" / "hardware"
MAP_DIR = ZIGZAG_DIR / "zigzag" / "inputs" / "mapping"

# Non-split ONNX workloads (MLPerfTiny quantised int8 models)
# ad_quantized_int8_conv.onnx is the AD model with QLinearMatMul converted to
# pointwise QLinearConv (see marp.onnx_tools.qlinearmatmul_to_conv).
NON_SPLIT_MODELS = [
    "ad_quantized_int8_conv.onnx",
    "ks_quantized_int8.onnx",
    "ic_quantized_int8.onnx",
    "mbv2_cifar10_int8.onnx",
]

# Standard (non-IMC) accelerator + mapping pairs
STANDARD_ACCELERATORS = [
    ("eyeriss_like.yaml", "default.yaml"),
    ("tpu_like.yaml", "tpu_like.yaml"),
    ("ascend_like.yaml", "ascend_like.yaml"),
    ("edge_tpu_like.yaml", "edge_tpu_like.yaml"),
]

# IMC accelerator + mapping pairs
IMC_ACCELERATORS = [
    ("dimc.yaml", "default_imc.yaml"),
    ("aimc.yaml", "default_imc.yaml"),
]


# ---------------------------------------------------------------------------
# Standard accelerator tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name", NON_SPLIT_MODELS, ids=[m.removesuffix(".onnx") for m in NON_SPLIT_MODELS]
)
@pytest.mark.parametrize(
    "hw_map",
    STANDARD_ACCELERATORS,
    ids=[hw.removesuffix(".yaml") for hw, _ in STANDARD_ACCELERATORS],
)
def test_standard_accelerator(model_name: str, hw_map: tuple[str, str]) -> None:
    """Run a non-split ONNX model on a standard accelerator."""
    hw_file, map_file = hw_map
    workload = _inferred_model(str(MODEL_DIR / model_name))
    accelerator = str(HW_DIR / hw_file)
    mapping = str(MAP_DIR / map_file)

    energy, latency, cmes = get_hardware_performance_zigzag(
        workload=workload,
        accelerator=accelerator,
        mapping=mapping,
        opt="latency",
        dump_folder=f"outputs/test_zigzag/{hw_file}/{model_name}",
        loma_show_progress_bar=False,
    )

    assert energy > 0, f"Energy should be positive, got {energy}"
    assert latency > 0, f"Latency should be positive, got {latency}"
    assert len(cmes) > 0, "Should produce at least one cost-model evaluation"

    # Print for golden-value capture
    print(f"[RESULT] {hw_file} | {model_name}: energy={energy}, latency={latency}")


# ---------------------------------------------------------------------------
# IMC accelerator tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name", NON_SPLIT_MODELS, ids=[m.removesuffix(".onnx") for m in NON_SPLIT_MODELS]
)
@pytest.mark.parametrize(
    "hw_map",
    IMC_ACCELERATORS,
    ids=[hw.removesuffix(".yaml") for hw, _ in IMC_ACCELERATORS],
)
def test_imc_accelerator(model_name: str, hw_map: tuple[str, str]) -> None:
    """Run a non-split ONNX model on an IMC accelerator."""
    hw_file, map_file = hw_map
    workload = _inferred_model(str(MODEL_DIR / model_name))
    accelerator = str(HW_DIR / hw_file)
    mapping = str(MAP_DIR / map_file)

    energy, latency, tclk, area, cmes = get_hardware_performance_zigzag(
        workload=workload,
        accelerator=accelerator,
        mapping=mapping,
        opt="latency",
        dump_folder=f"outputs/test_zigzag/{hw_file}/{model_name}",
        loma_show_progress_bar=False,
        in_memory_compute=True,
    )

    assert energy > 0, f"Energy should be positive, got {energy}"
    assert latency > 0, f"Latency should be positive, got {latency}"
    assert tclk > 0, f"Clock period should be positive, got {tclk}"
    assert area > 0, f"Area should be positive, got {area}"
    assert len(cmes) > 0, "Should produce at least one cost-model evaluation"

    # Print for golden-value capture
    print(
        f"[RESULT] {hw_file} | {model_name}: "
        f"energy={energy}, latency={latency}, tclk={tclk}, area={area}"
    )
