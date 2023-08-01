
#%%

import onnx
from onnx import numpy_helper as nphelp
import numpy as np
from torchvision import models,transforms
from PIL import Image
import matplotlib.pyplot as plt

from comp_graph import cgraph

print(f'Running ONNX version {onnx.__version__}')
nx_model = onnx.load('onnx_models/mbv2.onnx')

#%%

conv_node_UUT = cgraph.conv_node.from_onnx_node(nx_model,nx_model.graph.node[0])
relu_node_UUT = cgraph.clip_node.from_onnx_node(nx_model,nx_model.graph.node[3])
add_node_UUT = cgraph.add_node.from_onnx_node(nx_model,nx_model.graph.node[27])
dw_conv_UUT = cgraph.conv_node.from_onnx_depthwise(nx_model,nx_model.graph.node[4])
global_avg_node_UUT = cgraph.global_avg_node.from_onnx_node(nx_model,nx_model.graph.node[167])
flatten_node_UUT = cgraph.flatten_node.from_onnx_node(nx_model,nx_model.graph.node[168])
gemm_node_UUT = cgraph.gemm_node.from_onnx_node(nx_model,nx_model.graph.node[169])

#%%

img = Image.open('onnx_models/dog4.png')
img_tensor = transforms.ToTensor()(img).float()
tensor_input = img_tensor.unsqueeze(0)
img_array = np.array(img_tensor)

out = conv_node_UUT.forward(img_array)
out = relu_node_UUT.forward(out)

#%%

mbv2 = models.mobilenet_v2(pretrained=True)
mbv2.features[0][0].stride=(1,1)
mbv2.features[2].conv[1][0].stride=(1,1)

import torch
torch.onnx.export(mbv2,tensor_input,"mbv2.onnx")

mbv2.eval()
layer1_out = mbv2.features[0](tensor_input)

ch=1

plt.subplot(121,frameon=False,)
plt.imshow(out[ch])
plt.title('Output from my sim')
plt.subplot(122,frameon=False)
plt.imshow(layer1_out[0][ch].detach())
plt.title('Baseline torch output')

# %%
input = out
input_name = dw_conv_UUT[0][0].inputs[0]
outchs = []
for i,conv in enumerate(dw_conv_UUT[0]):
    if conv.inputs[0] != input_name:
        print(f'Input name anomaly found: {conv.inputs[0]} vs {input_name}')
    arr = conv.forward(np.expand_dims(input[i],0))
    outchs.append(arr.squeeze())
out2 = np.array(outchs)

# %%
import onnx
from onnx import numpy_helper as nphelp
import numpy as np
from torchvision import models,transforms
from PIL import Image
import matplotlib.pyplot as plt

from comp_graph import cgraph

print(f'Running ONNX version {onnx.__version__}')
nx_model = onnx.load('onnx_models/mbv2.onnx')

def onnx_to_cgraph(onnx_model):
    nodes = []
    for node in onnx_model.graph.node:
        # print(node.name)
        if node.op_type == 'Conv':
            if node.attribute[1].i == 1:
                nodes.append(cgraph.conv_node.from_onnx_node(onnx_model,node))
            else:
                convs,catter = cgraph.conv_node.from_onnx_depthwise(onnx_model,node)
                nodes.extend(convs)
                nodes.append(catter)    
        if node.op_type == 'Gemm':
            nodes.append(cgraph.gemm_node.from_onnx_node(onnx_model,node))
        if node.op_type == 'Add':
            nodes.append(cgraph.add_node.from_onnx_node(onnx_model,node))
        if node.op_type == 'Clip':
            nodes.append(cgraph.clip_node.from_onnx_node(onnx_model,node))
        if node.op_type == 'Flatten':
            nodes.append(cgraph.flatten_node.from_onnx_node(onnx_model,node))
        if node.op_type == 'GlobalAveragePool':
            nodes.append(cgraph.global_avg_node.from_onnx_node(onnx_model,node))

    return nodes

nodes = onnx_to_cgraph(nx_model)

img = Image.open('onnx_models/dog4.png')
img_tensor = transforms.ToTensor()(img).float()
tensor_input = img_tensor.unsqueeze(0)
img_array = np.array(img_tensor)

edges = dict()
for node in nodes:
    for input in node.inputs:
        edges[input] = None


def check_if_node_ready(node):
    for input in node.inputs:
        if edges[input] is None:
            return False
    return True

edges['input.1'] = img_array

from tqdm import tqdm

for node in tqdm(nodes):
    #print(node)
    check_if_node_ready(node)
    in_array = []
    for input in node.inputs:
        in_array.append(edges[input])
    # Squeeze eliminates extra dimension if there's only one input
    in_array = np.array(in_array).squeeze()
    out_array = node.forward(in_array)
    for output in node.outputs:
        edges[output] = out_array 

# TODO: We don't take into account strides... We need to.

# %%
