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
df = pd.DataFrame

# def test_import_bin_packing():
#     import bin_packing.hybrid_first_fit
#     import bin_packing.model_flattener
#     import bin_packing.objects

# def test_import_quantization():
#     import quantization.quant

@pytest.mark.skipif("not config.getoption('full')")
def test_cgraph_inference():
    modelpath = 'onnx_models/mobilenetv2-12-int8.onnx'

    nx_model = onnx.load(modelpath)
    img = Image.open('images/imagenet_finch.jpeg')
    img_tensor = transforms.ToTensor()(img).float()
    img_tensor = transforms.CenterCrop(224)(img_tensor)
    # tensor_input = img_tensor.unsqueeze()
    img_array = np.array(img_tensor)
    img_array = np.expand_dims(img_array, axis=0)

    input_dict = {
        'input':img_array
    }

    # ! rm -rf .cgraph_cache

    cgraph_UUT = cgraph.Cgraph.from_onnx_model(nx_model)
    out = cgraph_UUT.forward(input_dict,cachePath = '/home/lquizon/lawrence-workspace/SRAM_test/qrAcc2/qr_acc_2_digital/hwacc_design_garage/.cgraph_cache', recalculate=False) 
    top5 = np.argsort(out)[0][-5:]
    assert top5[-1] == 12

def test_onnx_inference():
    modelpath = 'onnx_models/mobilenetv2-12-int8.onnx'

    nx_model = onnx.load(modelpath)
    img = Image.open('images/imagenet_finch.jpeg')
    img_tensor = transforms.ToTensor()(img).float()
    img_tensor = transforms.CenterCrop(224)(img_tensor)
    # tensor_input = img_tensor.unsqueeze()
    img_array = np.array(img_tensor)
    img_array = np.expand_dims(img_array, axis=0)

    a = onnx_utils.get_intermediate_tensor_value(nx_model, "output", img_array)
    top5 = np.argsort(a)[0][-5:]
    assert top5[-1] == 12