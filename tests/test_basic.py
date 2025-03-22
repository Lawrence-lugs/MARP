from hwacctools.comp_graph import splitter, cnodes, cgraph
import numpy as np
import pytest
import hwacctools.onnx_utils as onnx_utils

def test_cgraph_inference(cgraph_uut,img_array):
    out = cgraph_uut.edges['output'] 
    assert np.squeeze(out).argmax() == 12

def test_onnx_inference(img_array):
    modelpath = 'onnx_models/mobilenetv2-12-int8.onnx'
    a = onnx_utils.get_intermediate_tensor_value(modelpath, "output", img_array)
    top5 = np.argsort(a)[0][-5:]
    assert top5[-1] == 12

def test_packed_cgraph_inference(core_packed,img_array):
    input_dict = {'input': img_array}
    out = core_packed.cgraph.forward(input_dict, recalculate=False) 
    assert np.squeeze(out).argmax() == 12

def test_hwc_inference(nx_model,img_array):    
    cgraph_UUT = cgraph.Cgraph.from_onnx_model(nx_model,cachePath = '.cgraph_cache', channel_minor=True)     
    input_dict = {'input': img_array}
    out = cgraph_UUT.forward(input_dict, recalculate=False)
    assert cgraph_UUT.nodes[1].channel_minor == True 
    assert np.squeeze(out).argmax() == 12