from hwacctools.comp_graph import cnodes, cgraph
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import hwacctools.onnx_utils as onnx_utils

@pytest.fixture
def qlinearconv_forward_output(cgraph_uut,nx_model,img_array):
    res_cg = cgraph_uut.edges['input_quantized'].astype(int)
    u_conv, u_scaler = cnodes.from_QLinearConv(nx_model,nx_model.graph.node[1])
    interm = u_conv.forward([res_cg])
    a = u_scaler.forward([interm])
    return a, interm

def test_from_qlinearconv(cgraph_uut,modelpath,input_dict):

    res = cgraph.compare_with_onnx(
        modelpath = modelpath,
        cgraph = cgraph_uut,
        input_tensor_name = 'input_quantized',
        output_tensor_name = '474_quantized',
        cgraph_input_dict = input_dict,
    )
    
    assert res

def test_from_qlinearmatmul(cgraph_uut,nx_model,modelpath,img_array):
    res_cg = onnx_utils.get_intermediate_tensor_value(modelpath, '472_quantized', img_array).astype(int)
    ref = onnx_utils.get_intermediate_tensor_value(modelpath, 'output_MatMul_quantized', img_array)
    u_conv, u_scaler = cnodes.from_QLinearMatMul(nx_model,nx_model.graph.node[-3])
    a = u_scaler.forward(u_conv.forward(res_cg))
    assert not (ref - a).any() 