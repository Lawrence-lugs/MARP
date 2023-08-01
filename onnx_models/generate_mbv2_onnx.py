#%%

import torch
from torchvision import models, transforms
import torch.onnx
import torchvision

import numpy as np
import matplotlib.pyplot as plt

mbv2 = models.mobilenet_v2(pretrained=True)
mbv2.features[0][0].stride=(1,1)
mbv2.features[2].conv[1][0].stride=(1,1)

#%%

from PIL import Image
dog = Image.open('onnx_models/dog4.png')
image_to_process = torchvision.transforms.ToTensor()(dog).float().unsqueeze(0)

mbv2.eval()
pred = mbv2(image_to_process)
layer1_out = mbv2.features[0][0](image_to_process)

# %%

torch.onnx.export(mbv2,image_to_process,"mbv2.onnx")
