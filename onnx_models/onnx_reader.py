#%%

import onnx
from onnx import numpy_helper as nphelp
import numpy as np

#%%

print(f'Running ONNX version {onnx.__version__}')
onnx_model = onnx.load('onnx_models/mbv2.onnx')


# 1. DONE Replace a convolution with matrices we like
# a. Concat Splits
# b. Add Splits
# 2. Perform an inference
# a. Flatten the inputs
# b. Rearrange the outputs
# 3. Loop over all the nodes, replacing the convolutions
# 4. Perform an inference over the entire network


def get_initializer_by_name(onnx_model,name):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return init
    raise LookupError

node = onnx_model.graph.node[0]
init = get_initializer_by_name(onnx_model,node.input[1])
weights = nphelp.to_array(init)

# Array shape is Och, Ich, K_W, K_H
# We need to flatten this into {Ich,K_w,K_h},Och

reshaped_weights = weights.reshape(weights.shape[0],-1)
biases = nphelp.to_array(get_initializer_by_name(onnx_model,node.input[2]))

#%%

# Next, we need to punt the dog flat

from PIL import Image
import matplotlib.pyplot as plt

dog = Image.open('onnx_models/dog4.png')
plt.imshow(dog)

dog = np.array(dog)

#%% 

# We need to define a function that flattens the picture in the way we want

def get_recfield_for_pixel(r,c,matrix):
    'Obtains the receptive field from the input'
    return matrix[r:r+3,c:c+3].transpose(2,0,1)

def toeplitzize_input(tensor):
    '''
    Assume tensor is in form H,W,C
    Also assume kernel is 3x3
    And zero padding
    '''
    H = tensor.shape[0]
    W = tensor.shape[1]
    C = tensor.shape[2]

    padded_tensor = np.pad(tensor,((1,1),(1,1),(0,0)))
    out = np.empty((H*W,C*9))

    for r,row in enumerate(tensor):
        for c,col in enumerate(row):
            out[r*W + c] = get_recfield_for_pixel(r,c,padded_tensor).flatten()

    return out

flat_dog = toeplitzize_input(dog)

#%%

flat_out = reshaped_weights @ flat_dog.T

# add the biases
for col in flat_out.T:
    col+=biases

# Now, we need to reconstruct a tensor out of the flat output

out = flat_out.reshape(32,32,-1)


# %% compare output with torch model

from torchvision import models, transforms

mbv2 = models.mobilenet_v2(pretrained=True)
mbv2.features[0][0].stride=(1,1)
mbv2.features[2].conv[1][0].stride=(1,1)

tensor_input = transforms.ToTensor()(image).float().unsqueeze(0)
layer1_out = mbv2.features[0][0](tensor_input)

plt.subplot(121,frameon=False)
plt.imshow(out[ch])
plt.title('Output from my sim')
plt.subplot(122,frameon=False)
plt.imshow(layer1_out[0][ch].detach())
plt.title('Baseline torch output')
