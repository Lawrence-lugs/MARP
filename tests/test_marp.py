import numpy as np
import pytest
import marp.onnx_tools.onnx_utils as onnx_utils
from marp.mapping import core
from marp.mapping import packer_utils as pu
import onnx
import matplotlib
import os
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from marp.compile import compile

@pytest.fixture(params=[
    'onnx_models/ad_quantized_int8.onnx',
    'onnx_models/ks_quantized_int8.onnx',
    'onnx_models/mbv2_cifar10_int8.onnx',
    'onnx_models/ic_quantized_int8.onnx'
])
def modelpath(request):
    return request.param

@pytest.fixture(params=[
    'Naive',
    'Dense',
    'Balanced',
    'WriteOptimized'
])
def packerName(request):
    return request.param

@pytest.fixture
def u_marped(
    modelpath,
    packerName,
    core_size=(256,256)
):

    packer = pu.get_packer_by_type(packerName)
    nx_model = onnx.load(modelpath)

    u_marped = core.NxModelMapping(nx_model, imc_core_size=core_size, packer=packer)
        
    return u_marped

@pytest.fixture
def u_model(u_marped):
    u_qracc = core.QRAccModel(
        u_marped,
        num_cores=1
    )
    return u_qracc

def test_plot_marp(u_marped):
    # Test plotting without blocking - uses Agg backend
    u_marped.plot()
    assert True  # If no exception is raised, the test passes
    
def test_plot_marp_save_to_file(u_marped, modelpath, packerName):
    # Extract model name (ad, ks, mbv2, ic) from modelpath
    basename = modelpath.split('/')[-1]
    modelname = basename.split('_')[0]  # e.g., 'ad' from 'ad_quantized_int8.onnx'
    # Compose output filename
    filename = f"{modelname}_{packerName}.png"
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)
    u_marped.plot(filepath=output_file)
    assert os.path.exists(output_file)  # Check that file was created

@pytest.mark.parametrize("modelpath", [
    # 'onnx_models/mbv2_cifar10_int8.onnx', # Remove testing compile of mbv2- this is a waste of resources
    'onnx_models/ad_quantized_int8.onnx',
    'onnx_models/ks_quantized_int8.onnx',
    'onnx_models/ic_quantized_int8.onnx'
])
@pytest.mark.parametrize("packerName", [
    'Naive',
    'Dense',
    'Balanced',
    'WriteOptimized'
])
def test_traverse_and_compile_nx_graph(modelpath, packerName):
    nx_model = onnx.load(modelpath)
    packer = pu.get_packer_by_type(packerName)
    input_name = nx_model.graph.input[0].name
    input_shape = [d.dim_value for d in nx_model.graph.input[0].type.tensor_type.shape.dim]
    if input_shape[0] > 1:
        nx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
        input_shape[0] = 1
    input_dict = {
        input_name: np.random.rand(*input_shape).astype(np.float32)
    }
    commands = compile.traverse_and_compile_nx_graph(
        nx_model      = nx_model,
        input_dict    = input_dict,
        imc_core_size = (256, 256),
        dwc_core_size = 32,
        packer        = packer
    )
    assert commands != []  # Ensure commands were generated