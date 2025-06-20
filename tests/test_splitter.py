import pytest
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from hwacctools.onnx_tools.onnx_splitter import split_qlinearconv_node_per_output_and_input_channel, split_qlinearconv_node_per_channel, split_qlinearconv_node_per_output_channel

def make_qlinearconv_node(K, C, with_bias=True, group=1, name="qconv"):
    # Inputs: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, bias (optional)
    inputs = [
        "x", "x_scale", "x_zp", "w", "w_scale", "w_zp", "y_scale", "y_zp"
    ]
    if with_bias:
        inputs.append("bias")
    node = helper.make_node(
        "QLinearConv",
        inputs=inputs,
        outputs=["out"],
        name=name,
        group=group
    )
    return node

def make_graph_and_inits(K, C, with_bias=True, group=1, w_scale_shape=None, w_zp_shape=None):
    # Create dummy initializers for all required inputs
    inits = []
    # x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, bias
    inits.append(numpy_helper.from_array(np.zeros([1, C, 4, 4], dtype=np.uint8), name="x"))
    inits.append(numpy_helper.from_array(np.array([0.1], dtype=np.float32), name="x_scale"))
    inits.append(numpy_helper.from_array(np.array([0], dtype=np.uint8), name="x_zp"))
    inits.append(numpy_helper.from_array(np.zeros([K, C, 3, 3], dtype=np.uint8), name="w"))
    if w_scale_shape is None:
        w_scale = np.array([0.2], dtype=np.float32)
    else:
        w_scale = np.full(w_scale_shape, 0.2, dtype=np.float32)
    inits.append(numpy_helper.from_array(w_scale, name="w_scale"))
    if w_zp_shape is None:
        w_zp = np.array([0], dtype=np.uint8)
    else:
        w_zp = np.zeros(w_zp_shape, dtype=np.uint8)
    inits.append(numpy_helper.from_array(w_zp, name="w_zp"))
    inits.append(numpy_helper.from_array(np.array([0.3], dtype=np.float32), name="y_scale"))
    inits.append(numpy_helper.from_array(np.array([0], dtype=np.uint8), name="y_zp"))
    if with_bias:
        inits.append(numpy_helper.from_array(np.zeros([K], dtype=np.int32), name="bias"))
    return inits

def make_graph(K, C, with_bias=True, group=1, w_scale_shape=None, w_zp_shape=None):
    node = make_qlinearconv_node(K, C, with_bias, group)
    inits = make_graph_and_inits(K, C, with_bias, group, w_scale_shape, w_zp_shape)
    graph = helper.make_graph(
        nodes=[node],
        name="test_graph",
        inputs=[],
        outputs=[helper.make_tensor_value_info("out", TensorProto.UINT8, None)],
        initializer=inits
    )
    return graph, node

def get_initializer(graph, name):
    for i in graph.initializer:
        if i.name == name:
            return i
    return None

def test_no_split_needed():
    K, C = 2, 2
    graph, node = make_graph(K, C)
    nodes, inits, out = split_qlinearconv_node_per_output_and_input_channel(graph, node, K_max=4, C_max=4)
    assert nodes == [node]
    assert inits == []
    assert out == node.output[0]

def test_split_output_channels():
    K, C = 5, 2
    graph, node = make_graph(K, C)
    nodes, inits, out = split_qlinearconv_node_per_output_channel(graph, node, K_max=4)
    # Should split into 2 QLinearConv nodes, then concat
    qconv_count = sum(1 for n in nodes if n.op_type == "QLinearConv")
    concat_count = sum(1 for n in nodes if n.op_type == "Concat")
    assert qconv_count == 2
    assert concat_count == 1
    assert out.endswith("concat_out")

def test_split_input_channels():
    K, C = 2, 5
    graph, node = make_graph(K, C)
    nodes, inits, out = split_qlinearconv_node_per_channel(graph, node, C_max=2)
    # Should split into 3 QLinearConv nodes (2+2+1), then QLinearAdd chain
    qconv_count = sum(1 for n in nodes if n.op_type == "QLinearConv")
    qadd_count = sum(1 for n in nodes if n.op_type == "QLinearAdd")
    assert qconv_count == 3
    assert qadd_count == 2
    assert out.startswith("split_add_2"), f"Output node name: {out}"

def test_split_both_channels():
    K, C = 5, 5
    graph, node = make_graph(K, C)
    nodes, inits, out = split_qlinearconv_node_per_output_and_input_channel(graph, node, K_max=2, C_max=2)
    # Should split K: 2+2+1, each with C: 2+2+1, so 3x3 QLinearConv, plus adds and concat
    qconv_count = sum(1 for n in nodes if n.op_type == "QLinearConv")
    concat_count = sum(1 for n in nodes if n.op_type == "Concat")
    assert qconv_count == 9
    assert concat_count == 1
    assert out.endswith("concat_out")

def test_depthwise_split():
    K, C = 4, 1
    group = 4
    graph, node = make_graph(K, C, group=group)
    nodes, inits, out = split_qlinearconv_node_per_output_and_input_channel(graph, node, K_max=2, C_max=1)
    # Should split as depthwise, so 2+2, with input slicing
    slice_count = sum(1 for n in nodes if n.op_type == "Slice")
    qconv_count = sum(1 for n in nodes if n.op_type == "QLinearConv")
    concat_count = sum(1 for n in nodes if n.op_type == "Concat")
    assert slice_count == 2
    assert qconv_count == 2
    assert concat_count == 1

def test_bias_absent():
    K, C = 3, 3
    graph, node = make_graph(K, C, with_bias=False)
    nodes, inits, out = split_qlinearconv_node_per_output_and_input_channel(graph, node, K_max=2, C_max=2)
    # Should not create bias initializers
    bias_inits = [i for i in inits if "bias" in i.name]
    assert len(bias_inits) == 0

def test_weight_scale_and_zp_per_channel():
    K, C = 4, 2
    graph, node = make_graph(K, C, w_scale_shape=(K,), w_zp_shape=(K,))
    nodes, inits, out = split_qlinearconv_node_per_output_and_input_channel(graph, node, K_max=2, C_max=2)
    # Should create split w_scale and w_zp initializers
    w_scale_inits = [i for i in inits if "w_scale" in i.name]
    w_zp_inits = [i for i in inits if "w_zp" in i.name]
    assert len(w_scale_inits) > 0
    assert len(w_zp_inits) > 0