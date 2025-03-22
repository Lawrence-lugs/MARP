from hwacctools.comp_graph import splitter, cnodes, cgraph
import onnx
from onnx import numpy_helper as nphelp
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import onnxruntime as ort
from onnx import helper
import hwacctools.onnx_utils as onnx_utils
import pandas as pd
import pytest

@pytest.fixture
def nx_model():
    modelpath = 'onnx_models/mobilenetv2-12-int8.onnx'
    nx_model = onnx.load(modelpath)
    return nx_model

@pytest.fixture
def img_array():
    img = Image.open('images/imagenet_finch.jpeg')
    img_tensor = transforms.ToTensor()(img).float()
    img_tensor = transforms.CenterCrop(224)(img_tensor)
    # tensor_input = img_tensor.unsqueeze()
    img_array = np.array(img_tensor)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@pytest.fixture
def cgraph_uut(nx_model,img_array):
    cgraph_UUT = cgraph.Cgraph.from_onnx_model(nx_model,cachePath = '.cgraph_cache') 
    return cgraph_UUT

def test_cgraph_inference(cgraph_uut,img_array):
    input_dict = {'input': img_array}
    out = cgraph_uut.forward(input_dict, recalculate=False) 
    assert np.squeeze(out).argmax() == 12

def test_onnx_inference(img_array):
    modelpath = 'onnx_models/mobilenetv2-12-int8.onnx'
    a = onnx_utils.get_intermediate_tensor_value(modelpath, "output", img_array)
    top5 = np.argsort(a)[0][-5:]
    assert top5[-1] == 12