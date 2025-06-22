import onnxruntime as ort
import numpy as np
from torchvision import transforms
from PIL import Image
import onnx
from onnx import helper, numpy_helper
from typing import Sequence, Any
import onnxruntime
from onnx.onnx_pb import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    TypeProto,
)

def add_tensor_to_model_outputs(model, tensor_name):
    layer_value_info = helper.ValueInfoProto()
    layer_value_info.name = tensor_name
    model.graph.output.append(layer_value_info)
    return model

def get_intermediate_tensor_value(modelpath, tensor_name, input_array = None, input_dict = None):
    if input_array is None and input_dict is None:
        raise ValueError("Either input_array or input_dict must be provided.")

    if input_dict is None:
        input_dict = {'input': input_array}         

    if type(modelpath) == str:
        model = onnx.load(modelpath)
    else :
        model = modelpath
    
    model = add_tensor_to_model_outputs(model, tensor_name)
    
    return infer(model, input_dict)[-1]

def infer(nx_model, input_dict):
    session = ort.InferenceSession(nx_model.SerializeToString())
    outputs = session.run(None, input_dict)
    return outputs

def _extract_value_info(
    input: list[Any] | np.ndarray | None,
    name: str,
    type_proto: TypeProto | None = None,
) -> onnx.ValueInfoProto:
    if type_proto is None:
        if input is None:
            raise NotImplementedError(
                "_extract_value_info: both input and type_proto arguments cannot be None."
            )
        elif isinstance(input, list):
            elem_type = onnx.helper.np_dtype_to_tensor_dtype(input[0].dtype)
            shape = None
            tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)
            type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
        elif isinstance(input, TensorProto):
            elem_type = input.data_type
            shape = tuple(input.dims)
            type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)
        else:
            elem_type = onnx.helper.np_dtype_to_tensor_dtype(input.dtype)
            shape = input.shape
            type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)

    return onnx.helper.make_value_info(name, type_proto)

def expect(
    node: onnx.NodeProto,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
    name: str,
    **kwargs: Any,
) -> None:
    # Builds the model
    present_inputs = [x for x in node.input if (x != "")]
    present_outputs = [x for x in node.output if (x != "")]
    input_type_protos = [None] * len(inputs)
    if "input_type_protos" in kwargs:
        input_type_protos = kwargs["input_type_protos"]
        del kwargs["input_type_protos"]
    output_type_protos = [None] * len(outputs)
    if "output_type_protos" in kwargs:
        output_type_protos = kwargs["output_type_protos"]
        del kwargs["output_type_protos"]
    inputs_vi = [
        _extract_value_info(arr, arr_name, input_type)
        for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)
    ]
    outputs_vi = [
        _extract_value_info(arr, arr_name, output_type)
        for arr, arr_name, output_type in zip(
            outputs, present_outputs, output_type_protos
        )
    ]
    graph = onnx.helper.make_graph(
        nodes=[node], name=name, inputs=inputs_vi, outputs=outputs_vi
    )
    kwargs["producer_name"] = "backend-test"

    if "opset_imports" not in kwargs:
        # To make sure the model will be produced with the same opset_version after opset changes
        # By default, it uses since_version as opset_version for produced models
        produce_opset_version = onnx.defs.get_schema(
            node.op_type, domain=node.domain
        ).since_version
        kwargs["opset_imports"] = [
            onnx.helper.make_operatorsetid(node.domain, produce_opset_version)
        ]

    model = onnx.helper.make_model_gen_version(graph, **kwargs)

    # Checking the produces are the expected ones.
    sess = onnxruntime.InferenceSession(model.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
    feeds = {name: value for name, value in zip(node.input, inputs)}
    results = sess.run(None, feeds)
    for expected, output in zip(outputs, results):
        np.allclose(expected, output)

def infer_node_output(
    node: onnx.NodeProto,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
    name: str,
    **kwargs: Any,
) -> None:
    # Builds the model
    present_inputs = [x for x in node.input if (x != "")]
    present_outputs = [x for x in node.output if (x != "")]
    input_type_protos = [None] * len(inputs)
    if "input_type_protos" in kwargs:
        input_type_protos = kwargs["input_type_protos"]
        del kwargs["input_type_protos"]
    output_type_protos = [None] * len(outputs)
    if "output_type_protos" in kwargs:
        output_type_protos = kwargs["output_type_protos"]
        del kwargs["output_type_protos"]
    inputs_vi = [
        _extract_value_info(arr, arr_name, input_type)
        for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)
    ]
    outputs_vi = [
        _extract_value_info(arr, arr_name, output_type)
        for arr, arr_name, output_type in zip(
            outputs, present_outputs, output_type_protos
        )
    ]
    graph = onnx.helper.make_graph(
        nodes=[node], name=name, inputs=inputs_vi, outputs=outputs_vi
    )
    kwargs["producer_name"] = "backend-test"

    if "opset_imports" not in kwargs:
        # To make sure the model will be produced with the same opset_version after opset changes
        # By default, it uses since_version as opset_version for produced models
        produce_opset_version = onnx.defs.get_schema(
            node.op_type, domain=node.domain
        ).since_version
        kwargs["opset_imports"] = [
            onnx.helper.make_operatorsetid(node.domain, produce_opset_version)
        ]

    model = onnx.helper.make_model_gen_version(graph, **kwargs)

    # Checking the produces are the expected ones.
    sess = onnxruntime.InferenceSession(model.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
    feeds = {name: value for name, value in zip(node.input, inputs)}
    results = sess.run(None, feeds)
    return results
    

def is_initializer(onnx_model,name):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return 'initializer'
    return False

def get_initializer_by_name(onnx_model,name):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return init
    raise LookupError(f'Could not find initializer with name {name}')

def get_node_by_output(onnx_model,output_name):
    for node in onnx_model.graph.node:
        if node.output[0] == output_name:
            return node
    raise LookupError(f'Could not find node with output {output_name}')

def get_attribute_by_name(name:str,attr_list:list):
    for i,attr in enumerate(attr_list):
        if attr.name == name:
            return attr
    raise AttributeError


def delete_initializer_by_name(model, initializer_name):
    for i, init in enumerate(model.graph.initializer):
        if init.name == initializer_name:
            del model.graph.initializer[i]
            break

def randomize_initializer_to_binary(model, initializer_name):
    array = numpy_helper.to_array(get_initializer_by_name(model, initializer_name))
    new_value = np.random.randint(0, 2, size=array.shape).astype(array.dtype)
    tensor = numpy_helper.from_array(new_value, name=initializer_name)
    delete_initializer_by_name(model, initializer_name)
    model.graph.initializer.append(tensor)

def randomize_model_to_binary_weights(model):
    for i,node in enumerate(model.graph.node):
        if node.op_type == 'QLinearConv':
            group = get_attribute_by_name('group', node.attribute).i
            if(group == 1):
                inii = node.input[3]
                randomize_initializer_to_binary(model, inii)
        if node.op_type == 'QLinearMatMul':
            inii = node.input[3]
            randomize_initializer_to_binary(model, inii)
    return model