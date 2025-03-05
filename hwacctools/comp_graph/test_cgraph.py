#%%

import onnx
from onnx import numpy_helper as nphelp
import numpy as np
from torchvision import models,transforms
from PIL import Image
import matplotlib.pyplot as plt
import splitter, cnodes, cgraph

print(f'Running ONNX version {onnx.__version__}')

nx_model = onnx.load('onnx_models/mbv2.onnx')

cgraph_UUT = cgraph.Cgraph.from_onnx_model(nx_model)

img = Image.open('onnx_models/dog4.png')
img_tensor = transforms.ToTensor()(img).float()
tensor_input = img_tensor.unsqueeze(0)
img_array = np.array(img_tensor)

input_dict = dict()
input_dict['input.1'] = img_array

out = cgraph_UUT.forward(input_dict)

def test_mbv2_cgraph():

    cgraph_predictions = np.array(out[0],dtype=float)
    onnx_predictions = np.load(r'C:\Users\Lawrence\lawrence-workspace\aimc-tasks\onnx_models\dog_norm_mbv2_outputs.npy').squeeze()

    # cgraph outputs same as onnx outputs
    assert np.allclose(cgraph_predictions,onnx_predictions,rtol=1e-2)

def test_conv_splitter():

    node_UUT = cgraph_UUT.nodes[-5]

    chx = splitter.split_conv_into_chunks(node_UUT,256,256)
    chx_UUT = cgraph.Cgraph(chx)
    test_key = node_UUT.inputs[0]
    test_array = cgraph_UUT.edges[node_UUT.inputs[0]]
    
    split_lastconv_fmap = chx_UUT.forward({test_key:test_array})
    split_lastconv_fmap = np.array(split_lastconv_fmap,dtype=float)

    # last convolution's outputs musts still be 
    assert np.allclose(split_lastconv_fmap,cgraph_UUT.edges[node_UUT.outputs[0]],rtol=1e-2)

def test_conv_splitter_feat8():
    '''
    Tests the problematic convolution at features[8] 

    This convolution is problematic because it splits into
    N rows & 1 column, making this an important test.
    '''
    
    node_UUT = None
    for node in cgraph_UUT.nodes:
        if node.outputs[0] == '/features/features.8/conv/conv.0/conv.0.0/Conv_output_0':
            node_UUT = node
    if node_UUT is None:
        raise LookupError

    chx = splitter.split_conv_into_chunks(node_UUT,256,256)
    chx_UUT = cgraph.Cgraph(chx)
    test_key = node_UUT.inputs[0]
    test_array = cgraph_UUT.edges[node_UUT.inputs[0]]
    
    split_lastconv_fmap = chx_UUT.forward({test_key:test_array})
    split_lastconv_fmap = np.array(split_lastconv_fmap,dtype=float)

    # last convolution's outputs musts still be 
    assert np.allclose(split_lastconv_fmap,cgraph_UUT.edges[node_UUT.outputs[0]],rtol=1e-2)


def test_gemm_splitter():

    node_UUT = cgraph_UUT.nodes[-1]
    cgraph_predictions = np.array(out[0],dtype=float)
    chx = splitter.split_gemm_into_chunks(node_UUT,256,256)

    chx_UUT = cgraph.Cgraph(chx)

    test_key = node_UUT.inputs[0]
    test_array = cgraph_UUT.edges[node_UUT.inputs[0]]
    
    split_gemm_logits = chx_UUT.forward({test_key:test_array})
    split_gemm_logits = np.array(split_gemm_logits,dtype=float)

    assert np.allclose(split_gemm_logits,cgraph_predictions,rtol=1e-2)

def test_cgraph_splitter():
    '''
    TODO: Parametrize this test
    '''

    W = 256
    H = 256

    split_cgraph_UUT = cgraph.split_convolutions(cgraph_UUT,H=H,W=W)

    for node in split_cgraph_UUT.nodes:
        if hasattr(node,'matrix'):
            assert node.matrix.shape[0] <= W
            assert node.matrix.shape[1] <= H

def test_split_mbv2_graph():
    '''
    TODO: Parametrize this test
    '''

    W = 256
    H = 256

    split_cgraph_UUT = cgraph.split_convolutions(cgraph_UUT,H=H,W=W)
    split_cgraph_logits = split_cgraph_UUT.forward(input_dict)
    split_cgraph_logits = np.array(split_cgraph_logits,dtype=float)
    
    onnx_predictions = np.load(r'C:\Users\Lawrence\lawrence-workspace\aimc-tasks\onnx_models\dog_norm_mbv2_outputs.npy').squeeze()
    
    assert np.allclose(split_cgraph_logits,onnx_predictions,rtol=1e-2)

# Core tests

import core

def test_acc_creation():
    '''
    TODO: Parametrize coresize
    '''
    core_size = (256,256)

    split_cgraph = cgraph.split_convolutions(cgraph_UUT,H=core_size[0],W=core_size[1])
    mbv2_system = core.Aimc_acc(split_cgraph,core_size)

    import hwacc_design_garage.hwacctools.bin_packing.packer_utils as packer_utils
    packer_utils.plot_packing_tiled(mbv2_system.packer,'mbv2_bssf')

    print(f'Number of cores: {len(mbv2_system.packer)}')
    return

