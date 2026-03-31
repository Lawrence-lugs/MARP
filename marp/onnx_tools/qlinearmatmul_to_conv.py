"""Convert QLinearMatMul nodes to pointwise QLinearConv in an ONNX model.

A QLinearMatMul ``y = x @ W`` with 2-D operands is mathematically equivalent
to a 1×1 (pointwise) QLinearConv when both x and W are reshaped to 4-D
tensors.  This script performs the conversion so that ZigZag (which only
supports Conv/QLinearConv) can analyse the model.

The conversion:

*  Input tensor  ``[N, Cin]``  → ``[N, Cin, 1, 1]``   (Reshape before conv)
*  Weight tensor ``[Cin, Cout]`` → ``[Cout, Cin, 1, 1]`` (transpose + reshape)
*  Output tensor ``[N, Cout, 1, 1]``  → ``[N, Cout]``  (Reshape after conv)

Usage::

    python -m marp.onnx_tools.qlinearmatmul_to_conv onnx_models/ad_quantized_int8.onnx
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _get_initializer(model: onnx.ModelProto, name: str) -> TensorProto | None:
    for init in model.graph.initializer:
        if init.name == name:
            return init
    return None


def _replace_initializer(model: onnx.ModelProto, name: str, new_arr: np.ndarray):
    """Replace an existing initializer tensor with *new_arr*."""
    for i, init in enumerate(model.graph.initializer):
        if init.name == name:
            new_init = numpy_helper.from_array(new_arr, name=name)
            model.graph.initializer[i].CopyFrom(new_init)
            return
    # Not found – add it
    model.graph.initializer.append(numpy_helper.from_array(new_arr, name=name))


def convert_qlinearmatmul_to_conv(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a *new* model with every QLinearMatMul replaced by a pointwise QLinearConv.

    Surrounding Reshape nodes are inserted so that the rest of the graph
    (which uses 2-D tensors) is unaffected.
    """
    model = copy.deepcopy(model)

    nodes_to_remove: list[onnx.NodeProto] = []
    nodes_to_add: list[onnx.NodeProto] = []

    for node in model.graph.node:
        if node.op_type != "QLinearMatMul":
            continue

        # QLinearMatMul inputs:
        # 0: a,  1: a_scale,  2: a_zero_point,
        # 3: b,  4: b_scale,  5: b_zero_point,
        # 6: y_scale,  7: y_zero_point
        a_name = node.input[0]
        a_scale = node.input[1]
        a_zp = node.input[2]
        b_name = node.input[3]
        b_scale = node.input[4]
        b_zp = node.input[5]
        y_scale = node.input[6]
        y_zp = node.input[7]
        y_name = node.output[0]

        # --- Reshape weight: [Cin, Cout] → [Cout, Cin, 1, 1] ---
        b_init = _get_initializer(model, b_name)
        if b_init is None:
            raise ValueError(
                f"QLinearMatMul node {node.name}: weight tensor '{b_name}' "
                f"is not an initializer — dynamic weights are not supported."
            )
        b_arr = numpy_helper.to_array(b_init)
        if b_arr.ndim != 2:
            raise ValueError(
                f"QLinearMatMul node {node.name}: expected 2-D weight, got shape {b_arr.shape}"
            )
        Cin, Cout = b_arr.shape
        # Conv weight layout: [out_channels, in_channels/group, kH, kW]
        b_conv = b_arr.T.reshape(Cout, Cin, 1, 1)
        _replace_initializer(model, b_name, b_conv)

        # --- Insert Reshape: [N, Cin] → [N, Cin, 1, 1] before conv ---
        reshape_in_shape_name = f"{node.name}__reshape_in_shape"
        reshape_in_out = f"{a_name}__4d"
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.array([0, 0, 1, 1], dtype=np.int64),
                name=reshape_in_shape_name,
            )
        )
        nodes_to_add.append(
            helper.make_node(
                "Reshape",
                inputs=[a_name, reshape_in_shape_name],
                outputs=[reshape_in_out],
                name=f"{node.name}__reshape_in",
            )
        )

        # --- QLinearConv (pointwise, 1×1, no padding, stride 1) ---
        conv_out = f"{y_name}__4d"
        conv_node = helper.make_node(
            "QLinearConv",
            inputs=[
                reshape_in_out,  # x
                a_scale,         # x_scale
                a_zp,            # x_zero_point
                b_name,          # w (now [Cout, Cin, 1, 1])
                b_scale,         # w_scale
                b_zp,            # w_zero_point
                y_scale,         # y_scale
                y_zp,            # y_zero_point
            ],
            outputs=[conv_out],
            name=f"{node.name}__conv",
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        )
        nodes_to_add.append(conv_node)

        # --- Insert Reshape: [N, Cout, 1, 1] → [N, Cout] after conv ---
        reshape_out_shape_name = f"{node.name}__reshape_out_shape"
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.array([0, -1], dtype=np.int64),
                name=reshape_out_shape_name,
            )
        )
        nodes_to_add.append(
            helper.make_node(
                "Reshape",
                inputs=[conv_out, reshape_out_shape_name],
                outputs=[y_name],
                name=f"{node.name}__reshape_out",
            )
        )

        nodes_to_remove.append(node)

    if not nodes_to_remove:
        return model

    # Apply graph surgery: remove old, insert new (preserving order)
    for old_node in nodes_to_remove:
        idx = list(model.graph.node).index(old_node)
        model.graph.node.remove(old_node)
        for j, new_node in enumerate(
            [n for n in nodes_to_add if n.name.startswith(old_node.name)]
        ):
            model.graph.node.insert(idx + j, new_node)

    return model


def main(input_path: str, output_path: str | None = None):
    model = onnx.load(input_path)
    new_model = convert_qlinearmatmul_to_conv(model)

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_stem(p.stem + "_conv"))

    onnx.save(new_model, output_path)
    print(f"Saved converted model to {output_path}")

    # Quick validation
    import onnxruntime as ort

    sess_old = ort.InferenceSession(onnx.load(input_path).SerializeToString())
    sess_new = ort.InferenceSession(new_model.SerializeToString())
    inp = sess_old.get_inputs()[0]
    dummy = np.random.randn(*inp.shape).astype(np.float32)
    out_old = sess_old.run(None, {inp.name: dummy})[0]
    out_new = sess_new.run(None, {inp.name: dummy})[0]
    max_diff = np.max(np.abs(out_old.astype(float) - out_new.astype(float)))
    print(f"Max output difference: {max_diff}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python -m marp.onnx_tools.qlinearmatmul_to_conv <input.onnx> [output.onnx]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
