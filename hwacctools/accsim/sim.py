import numpy as np
from hwacctools.comp_graph import cgraph, core
import onnx
import numpy as np
from torchvision import transforms
from PIL import Image
import hwacctools.onnx_utils as onnx_utils
import pandas as pd
df = pd.DataFrame

def generate_input_array(img_path):
    img = Image.open(img_path)
    img_tensor = transforms.ToTensor()(img).float()
    img_tensor = transforms.CenterCrop(224)(img_tensor)
    img_array = np.array(img_tensor)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def run_simulation():
    modelpath = 'onnx_models/mobilenetv2-12-int8.onnx'
    img_path = 'images/imagenet_finch.jpeg'
    nx_model = onnx.load(modelpath)
    # img_array = generate_input_array(img_path)
    # input_dict = {
    #     'input':img_array
    # }
    cgraph_UUT = cgraph.Cgraph.from_onnx_model(nx_model)
    u_packed = core.packed_model(cgraph_UUT,core_size=(256,256))
    
