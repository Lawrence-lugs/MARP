import onnxruntime as ort
import numpy as np
from torchvision import transforms
from PIL import Image
import onnx
from onnx import helper

def add_tensor_to_model_outputs(model, tensor_name):
    layer_value_info = helper.ValueInfoProto()
    layer_value_info.name = tensor_name
    model.graph.output.append(layer_value_info)
    return model

def get_intermediate_tensor_value(model, tensor_name, img_array):
    model = add_tensor_to_model_outputs(model, tensor_name)
    # The desired tensor is the last one in the outputs list
    return infer(model, img_array)[-1]

def infer(nx_model, img_array):
    session = ort.InferenceSession(nx_model.SerializeToString())
    outputs = session.run(None, {"input": img_array})
    return outputs