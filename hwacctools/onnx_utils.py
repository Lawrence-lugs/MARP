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

def get_intermediate_tensor_value(modelpath, tensor_name, input_array = None, input_dict = None):
    if input_array is None and input_dict is None:
        raise ValueError("Either input_array or input_dict must be provided.")

    if input_dict is None:
        input_dict = {'input': input_array}         

    if type(modelpath) == str:
        model = onnx.load(modelpath)
    else :
        model = modelpath.SerializeToString()
    
    model = add_tensor_to_model_outputs(model, tensor_name)
    
    return infer(model, input_dict)[-1]

def infer(nx_model, input_dict):
    session = ort.InferenceSession(nx_model.SerializeToString())
    outputs = session.run(None, input_dict)
    return outputs