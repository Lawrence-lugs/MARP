import onnx
from onnx import helper, numpy_helper, TensorProto

def split_qlinearconv_node_per_channel(graph, node, C_max, prefix="split"):
    """
    Splits a QLinearConv node into multiple QLinearConv nodes with at most C_max input channels,
    then sums their outputs with QLinearAdd nodes. Handles per-channel quantization for weights.
    
    Args:
        graph: The ONNX graph object.
        node: The QLinearConv node to split.
        C_max: Maximum number of input channels per split.
        prefix: Prefix for new node names.
    Returns:
        List of new nodes (QLinearConv and QLinearAdd), new initializers, and final output name.
    """
    # Find weight, scale, and zero point initializers
    weight_name = node.input[3]
    weight_scale_name = node.input[4]
    weight_zp_name = node.input[5]
    weight_init = next(i for i in graph.initializer if i.name == weight_name)
    weight_scale_init = next(i for i in graph.initializer if i.name == weight_scale_name)
    weight_zp_init = next(i for i in graph.initializer if i.name == weight_zp_name)
    W = numpy_helper.to_array(weight_init)
    W_scale = numpy_helper.to_array(weight_scale_init)
    W_zp = numpy_helper.to_array(weight_zp_init)
    C_in = W.shape[1]  # (out_channels, in_channels, kH, kW)
    splits = []
    new_nodes = []
    new_inits = []
    out_names = []

    for i in range(0, C_in, C_max):
        c_start = i
        c_end = min(i + C_max, C_in)
        # Slice weights
        W_split = W[:, c_start:c_end, :, :]
        w_split_name = f"{prefix}_w_{i}"
        w_split_init = numpy_helper.from_array(W_split, name=w_split_name)
        new_inits.append(w_split_init)

        # For per-channel quantization, weight scale and zp are per out_channel
        # So we use the same scale/zp for each split (no slicing needed)
        # If per-channel quantization is on input channels, slice accordingly

        # Create new QLinearConv node
        inputs = list(node.input)
        inputs[3] = w_split_name  # Replace weight input
        out_name = f"{prefix}_out_{i}"
        out_names.append(out_name)
        qconv_node = helper.make_node(
            "QLinearConv",
            inputs=inputs,
            outputs=[out_name],
            name=f"{prefix}_qconv_{i}"
        )
        new_nodes.append(qconv_node)

    # Sum outputs with QLinearAdd nodes
    sum_out = out_names[0]
    for i in range(1, len(out_names)):
        add_out = f"{prefix}_add_{i}"
        qadd_node = helper.make_node(
            "QLinearAdd",
            # QLinearAdd: A, B, a_scale, a_zero_point, b_scale, b_zero_point, y_scale, y_zero_point
            inputs=[
                sum_out, 
                node.input[6], node.input[7],  # a_scale, a_zero_point (use output scale/zp)
                out_names[i],
                node.input[6], node.input[7],  # b_scale, b_zero_point (use output scale/zp)
                node.input[6], node.input[7],  # y_scale, y_zero_point (use output scale/zp)
            ],
            outputs=[add_out],
            name=f"{prefix}_qadd_{i}",
            domain="com.microsoft"  # Use Microsoft domain for QLinearAdd
        )
        new_nodes.append(qadd_node)
        sum_out = add_out

    return new_nodes, new_inits, sum_out

def replace_qlinearconv_with_split(graph, node, new_nodes, new_inits, final_output):
    """
    Replaces a QLinearConv node in the graph with a set of new nodes and initializers.
    The output of the last new node will be connected to the original node's output.

    Args:
        graph: The ONNX graph object.
        node: The QLinearConv node to replace.
        new_nodes: List of new nodes (QLinearConv and QLinearAdd).
        new_inits: List of new initializers.
        final_output: Name of the final output tensor from the split nodes.
    """
    # Remove the original node
    node_idx = None
    for idx, n in enumerate(graph.node):
        if n == node:
            node_idx = idx
            break
    if node_idx is not None:
        del graph.node[node_idx]

    # Add new initializers
    for init in new_inits:
        graph.initializer.append(init)

    # Add new nodes
    for n in new_nodes:
        graph.node.append(n)

    # Redirect outputs: find all nodes that use the original output and update their input
    orig_output = node.output[0]
    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp == orig_output:
                n.input[i] = final_output

    # If the graph output is the original node's output, update it too
    for out in graph.output:
        if out.name == orig_output:
            out.name = final_output

def split_qlinearconv_node_per_output_channel(graph, node, K_max, prefix="split_out"):
    """
    Splits a QLinearConv node into multiple QLinearConv nodes with at most K_max output channels,
    then concatenates their outputs to emulate the original node.

    Args:
        graph: The ONNX graph object.
        node: The QLinearConv node to split.
        K_max: Maximum number of output channels per split.
        prefix: Prefix for new node names.
    Returns:
        List of new nodes (QLinearConv and Concat), new initializers, and final output name.
    """
    weight_name = node.input[3]
    weight_scale_name = node.input[4]
    weight_zp_name = node.input[5]
    weight_init = next(i for i in graph.initializer if i.name == weight_name)
    weight_scale_init = next(i for i in graph.initializer if i.name == weight_scale_name)
    weight_zp_init = next(i for i in graph.initializer if i.name == weight_zp_name)
    W = numpy_helper.to_array(weight_init)
    W_scale = numpy_helper.to_array(weight_scale_init)
    W_zp = numpy_helper.to_array(weight_zp_init)
    K = W.shape[0]  # (out_channels, in_channels, kH, kW)

    new_nodes = []
    new_inits = []
    out_names = []

    for i in range(0, K, K_max):
        k_start = i
        k_end = min(i + K_max, K)
        # Slice weights and quant params
        W_split = W[k_start:k_end, :, :, :]
        w_split_name = f"{prefix}_w_{i}"
        w_split_init = numpy_helper.from_array(W_split, name=w_split_name)
        new_inits.append(w_split_init)

        # Slice per-channel quantization params for weights
        if W_scale.ndim == 1 and len(W_scale) == K:
            W_scale_split = W_scale[k_start:k_end]
            w_scale_split_name = f"{prefix}_w_scale_{i}"
            w_scale_split_init = numpy_helper.from_array(W_scale_split, name=w_scale_split_name)
            new_inits.append(w_scale_split_init)
        else:
            w_scale_split_name = weight_scale_name  # fallback to original

        if W_zp.ndim == 1 and len(W_zp) == K:
            W_zp_split = W_zp[k_start:k_end]
            w_zp_split_name = f"{prefix}_w_zp_{i}"
            w_zp_split_init = numpy_helper.from_array(W_zp_split, name=w_zp_split_name)
            new_inits.append(w_zp_split_init)
        else:
            w_zp_split_name = weight_zp_name  # fallback to original

        # Create new QLinearConv node
        inputs = list(node.input)
        inputs[3] = w_split_name
        inputs[4] = w_scale_split_name
        inputs[5] = w_zp_split_name
        out_name = f"{prefix}_out_{i}"
        out_names.append(out_name)
        qconv_node = helper.make_node(
            "QLinearConv",
            inputs=inputs,
            outputs=[out_name],
            name=f"{prefix}_qconv_{i}"
        )
        new_nodes.append(qconv_node)

    # Concatenate outputs along output channel axis (axis=1 for NCHW)
    concat_out = f"{prefix}_concat_out"
    concat_node = helper.make_node(
        "Concat",
        inputs=out_names,
        outputs=[concat_out],
        name=f"{prefix}_concat",
        axis=1
    )
    new_nodes.append(concat_node)

    return new_nodes, new_inits, concat_out

def split_qlinearconv_node_per_output_and_input_channel(graph, node, K_max, C_max, prefix="split_both"):
    """
    Splits a QLinearConv node into multiple QLinearConv nodes with at most K_max output channels
    and at most C_max input channels, using both output and input channel splitting.
    The outputs are summed and concatenated to emulate the original node.

    Args:
        graph: The ONNX graph object.
        node: The QLinearConv node to split.
        K_max: Maximum number of output channels per split.
        C_max: Maximum number of input channels per split.
        prefix: Prefix for new node names.
    Returns:
        List of new nodes, new initializers, and final output name.
    """
    weight_name = node.input[3]
    weight_init = next(i for i in graph.initializer if i.name == weight_name)
    W = numpy_helper.to_array(weight_init)
    K = W.shape[0]  # out_channels

    # if both dimensions are less than their respective maxes, we can skip splitting
    if W.shape[0] <= K_max and W.shape[1] <= C_max:
        return [node], [], node.output[0]

    # if W.shape[0] is less than K_max, we can skip splitting by output channels
    if K <= K_max:
        return split_qlinearconv_node_per_channel(graph, node, C_max, prefix)
    
    # if W.shape[1] is less than C_max, we can skip splitting by input channels
    if W.shape[1] <= C_max:
        return split_qlinearconv_node_per_output_channel(graph, node, K_max, prefix)    

    all_nodes = []
    all_inits = []
    concat_outputs = []

    for k_idx, k_start in enumerate(range(0, K, K_max)):
        k_end = min(k_start + K_max, K)
        # Slice node for output channels
        # Create a temporary node with sliced weights and quant params for output channels
        # Use split_qlinearconv_node_per_output_channel to get nodes for this output slice
        # But we want to further split each output-sliced QLinearConv by input channels

        # Prepare a temporary node with output channel slice
        # We'll need to slice weights, scales, and zero points for this output channel range
        weight_scale_name = node.input[4]
        weight_zp_name = node.input[5]
        weight_scale_init = next(i for i in graph.initializer if i.name == weight_scale_name)
        weight_zp_init = next(i for i in graph.initializer if i.name == weight_zp_name)
        W_scale = numpy_helper.to_array(weight_scale_init)
        W_zp = numpy_helper.to_array(weight_zp_init)

        # Slice weights and quant params for output channels
        W_k = W[k_start:k_end, :, :, :]
        w_k_name = f"{prefix}_w_{k_idx}"
        w_k_init = numpy_helper.from_array(W_k, name=w_k_name)
        all_inits.append(w_k_init)

        if W_scale.ndim == 1 and len(W_scale) == K:
            W_scale_k = W_scale[k_start:k_end]
            w_scale_k_name = f"{prefix}_w_scale_{k_idx}"
            w_scale_k_init = numpy_helper.from_array(W_scale_k, name=w_scale_k_name)
            all_inits.append(w_scale_k_init)
        else:
            w_scale_k_name = weight_scale_name

        if W_zp.ndim == 1 and len(W_zp) == K:
            W_zp_k = W_zp[k_start:k_end]
            w_zp_k_name = f"{prefix}_w_zp_{k_idx}"
            w_zp_k_init = numpy_helper.from_array(W_zp_k, name=w_zp_k_name)
            all_inits.append(w_zp_k_init)
        else:
            w_zp_k_name = weight_zp_name

        # Create a temporary node for this output channel slice
        temp_inputs = list(node.input)
        temp_inputs[3] = w_k_name
        temp_inputs[4] = w_scale_k_name
        temp_inputs[5] = w_zp_k_name

        temp_node = helper.make_node(
            "QLinearConv",
            inputs=temp_inputs,
            outputs=[f"{prefix}_tmp_out_{k_idx}"],
            name=f"{prefix}_tmp_qconv_{k_idx}"
        )

        # Now split this temp node by input channels using split_qlinearconv_node_per_channel
        # We need to update the weight initializer in the graph for this temp node
        # But instead, we can pass the sliced weights and quant params directly to the splitting function
        # So, we create a temporary graph with only this temp node and its inits
        # Instead, let's call split_qlinearconv_node_per_channel logic directly here

        # --- Begin input channel splitting ---
        W_k_C = W_k
        C_in = W_k_C.shape[1]
        out_names = []
        nodes = []
        inits = []

        for c_idx, c_start in enumerate(range(0, C_in, C_max)):
            c_end = min(c_start + C_max, C_in)
            W_split = W_k_C[:, c_start:c_end, :, :]
            w_split_name = f"{prefix}_w_{k_idx}_{c_idx}"
            w_split_init = numpy_helper.from_array(W_split, name=w_split_name)
            inits.append(w_split_init)

            # Use the same per-output-channel quant params for each input split
            # (per ONNX QLinearConv spec)
            out_name = f"{prefix}_out_{k_idx}_{c_idx}"
            out_names.append(out_name)
            qconv_inputs = list(temp_inputs)
            qconv_inputs[3] = w_split_name
            # scale/zp already set in temp_inputs

            qconv_node = helper.make_node(
                "QLinearConv",
                inputs=qconv_inputs,
                outputs=[out_name],
                name=f"{prefix}_qconv_{k_idx}_{c_idx}"
            )
            nodes.append(qconv_node)

        # Sum all input-channel-split outputs for this output channel slice
        sum_out = out_names[0]
        for i in range(1, len(out_names)):
            add_out = f"{prefix}_add_{k_idx}_{i}"
            qadd_node = helper.make_node(
                "QLinearAdd",
                inputs=[
                    sum_out, 
                    node.input[6], node.input[7],  # a_scale, a_zero_point (use output scale/zp)
                    out_names[i],
                    node.input[6], node.input[7],  # b_scale, b_zero_point (use output scale/zp)
                    node.input[6], node.input[7],  # y_scale, y_zero_point (use output scale/zp)
                ],
                outputs=[add_out],
                name=f"{prefix}_qadd_{k_idx}_{i}"
            )
            nodes.append(qadd_node)
            sum_out = add_out

        all_nodes.extend(nodes)
        all_inits.extend(inits)
        concat_outputs.append(sum_out)
        # --- End input channel splitting ---

    # Concatenate all output-channel-split results
    concat_out = f"{prefix}_concat_out"
    concat_node = helper.make_node(
        "Concat",
        inputs=concat_outputs,
        outputs=[concat_out],
        name=f"{prefix}_concat",
        axis=1
    )
    all_nodes.append(concat_node)

    return all_nodes, all_inits, concat_out

def split_model_to_per_channel(graph, C_max=256, K_max=256, prefix="split"):
    """
    Applies the QLinearConv splitters to every QLinearConv node in the ONNX graph,
    limiting each to at most K_max output channels and C_max input channels.

    Args:
        graph: The ONNX graph object.
        C_max: Maximum number of input channels per split.
        K_max: Maximum number of output channels per split.
        prefix: Prefix for new node names.
    """
    # Collect QLinearConv nodes first to avoid modifying the list while iterating
    qconv_nodes = [node for node in graph.node if node.op_type == "QLinearConv"]

    for idx, node in enumerate(qconv_nodes):
        # Split the node
        new_nodes, new_inits, final_output = split_qlinearconv_node_per_output_and_input_channel(
            graph, node, K_max, C_max, prefix=f"{prefix}_{idx}"
        )

        if len(new_nodes) == 0:
            continue

        # Replace the original node with the split nodes
        replace_qlinearconv_with_split(graph, node, new_nodes, new_inits, final_output)