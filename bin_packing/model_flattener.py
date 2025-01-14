# Flattens torch model into matrices for bin packing

import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def remove_sequential(network):
    all_layers = []
    for layer in network.children():
        if list(layer.children()) == []: # if leaf node, add it to list
            all_layers.append(layer)
        else:
            all_layers.extend(remove_sequential(layer))
    return all_layers

def print_model_layertypes(layerset):
    layertypes = []
    for layer in layerset:
        if type(layer) not in layertypes:
            layertypes.append(type(layer))
    print(layertypes)

def matricize_conv(layer):
    mat = torch.flatten(layer.weight,start_dim=1)
    mat = mat.cpu().detach().numpy()
    return mat

def matricize_gconv(layer):
    matin = matricize_conv(layer)
    g = layer.groups
    if g == 1:
        return matin
    kernel_shape = np.shape(matin)[1]
    kernels_per_group = int(np.shape(matin)[0]/g)
    matin = np.reshape(matin,(g,kernels_per_group,kernel_shape))
    mat = []
    for k in np.arange(g):
        mat.append(matin[k])
    return np.array(mat)

def matricize_linear(layer):
    mat = layer.weight
    mat = mat.cpu().detach().numpy()
    mat = np.append(mat,layer.bias.cpu()[:,None],axis=1)
    return mat

def apply_bn(mat,bnlayer):
    bnlayer = bnlayer.cpu()
    a,b,c=0,0,0
    mat2 = np.copy(mat)
    if mat.ndim == 3:
        #print(mat.shape)
        a,b,c = mat.shape
        mat = np.reshape(mat,(a*b,c))
    for i,ch in enumerate(mat):
        mat[i] = bnlayer.weight[i] * mat[i]
    mat = np.append(mat,bnlayer.bias[:,None],axis=1)
    if mat2.ndim == 3:
        mat = np.reshape(mat,(a,b,c+1))
    return mat
        
def matricize_model(model):
    layer_list = remove_sequential(model)
    mat_set = []
    strides = []
    prev_mat = 0
    for i,layer in enumerate(layer_list):
        if type(layer)==nn.Conv2d:
            mat_set.append(matricize_gconv(layer))
            strides.append(layer.stride[0])
        if type(layer)==nn.BatchNorm2d:
            # apply to last seen convolution
            mat_set[-1] = apply_bn(mat_set[-1],layer)
        if type(layer)==nn.Linear:
            mat_set.append(matricize_linear(layer))
        #skip activations
    return strides,mat_set

def get_shapes(flatmodel):
    model_shapes = []
    for mat in flatmodel:
        model_shapes.append(mat.shape)
    return model_shapes

def count_parameters(flatmodel):
    shapes = get_shapes(flatmodel)
    out = 0
    for shape in shapes:
        out += shape[0]*shape[1]
    return out

if __name__ == '__main__':

    import torchvision

    mbv2 = torchvision.models.mobilenet_v2(pretrained=True)

    with torch.no_grad():
        K,mbv2_flat = matricize_model(mbv2)

    mbv2_shapes = get_shapes(mbv2_flat)
    print(mbv2_shapes)

# %%
