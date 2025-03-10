import onnx
from onnx import numpy_helper as nphelp
import numpy as np
from .compute import *
from ..quantization import quant as q
from joblib import Memory
from cnodes import *

def is_initializer(onnx_model,name):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return 'initializer'
    return False

def get_initializer_by_name(onnx_model,name):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return init
    raise LookupError(f'Could not find initializer with name {name}')

def get_node_by_output(onnx_model,output_name):
    for node in onnx_model.graph.node:
        if node.output[0] == output_name:
            return node
    raise LookupError(f'Could not find node with output {output_name}')

def get_attribute_by_name(name:str,attr_list:list):
    for i,attr in enumerate(attr_list):
        if attr.name == name:
            return attr
    raise AttributeError

class output_scale_node_hwlike(Node):
    def __init__(self, inputs, outputs, scale_x, scale_w, scale_y, scale_precision = 16, out_precision=8):
        '''
        Output scaling per TFLite quantization
        '''
        super(output_scale_node,self).__init__(inputs,outputs)
        
        newscale = scale_x * scale_w / scale_y

        self.real_scale = newscale

        m0, shift = q.vconvert_scale_to_shift_and_m0(
                        newscale,
                        precision=scale_precision
        )

        fp_m = m0 * np.power(2.,shift)

        self.m0 = m0
        self.shift = shift
        self.fp_m = fp_m

        self.out_precision = out_precision
        self.scale_precision = scale_precision
    
    def forward(self,input:np.array):
        input_squeezed = np.array(input).squeeze()
        # black magic needed to force fp_m multiplication to broadcast along the first dimension of out (C)
        fp_m = self.fp_m
        fp_m = fp_m.reshape(fp_m.shape[0],*([1]*(len(input_squeezed.shape)-1)))
        out = (input_squeezed * fp_m).astype(int)
        out = q.saturating_clip(out, self.out_precision)

        # add back the batch axis
        out = out.reshape(1,*out.shape)
        return out
