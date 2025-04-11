import onnx
from onnx import numpy_helper as nphelp
import numpy as np
from .compute import *
from ..quantization import quant as q
from joblib import Memory

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

def get_cnode_from_onnx_node(node,nx_model, **kwargs):
    if node.op_type == 'Conv':
        if node.attribute[1].i == 1:
            return(conv_node.from_onnx_node(nx_model,node))
        else:
            convs,catter = conv_node.from_onnx_depthwise(nx_model,node)
            return(convs)
            return(catter)    
    elif node.op_type == 'Gemm':
        return(gemm_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'Add':
        return(add_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'Clip':
        return(clip_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'Flatten':
        return(flatten_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'GlobalAveragePool':
        return(global_avg_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'QuantizeLinear':
        return(quantize_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'QLinearConv':
        return(from_QLinearConv(nx_model,node,**kwargs))
    elif node.op_type == 'QLinearAdd':
        return(quantized_linear_add_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'QLinearGlobalAveragePool':
        return(quantized_global_avg_pool_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'DequantizeLinear':
        return(dequantize_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'Shape':
        return(shape_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'Reshape':
        return(reshape_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'Gather':
        return(gather_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'Unsqueeze':
        return(unsqueeze_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'Squeeze':
        return(squeeze_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'Concat':
        return(concat_node.from_onnx_node(nx_model,node))
    elif node.op_type == 'QLinearMatMul':
        return(from_QLinearMatMul(nx_model,node))
    elif node.op_type == 'QLinearAveragePool': # For now, all average pools are global
        return(quantized_global_avg_pool_node.from_onnx_node(nx_model,node))
    else:
        raise NotImplementedError(f'Node type {node.op_type} not implemented')

class Node(object):
    '''
    Generic node.
    Unimplemented types go here.

    My nodes tend to just be convenient surrogates for the heavy onnx nodes.
    '''
    def __init__(self, inputs:list[str], outputs:list[str], function = 'defined'):
        self.inputs = inputs 
        self.outputs = outputs
        self.function = function

    def forward(self):
        raise NotImplementedError
    
    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        
        inputs = [onnx_node.input[0]]
        outputs = onnx_node.output
        function = onnx_node.op_type

        return Node(inputs,outputs,function)

class reshape_node(Node):
    '''
    Reshapes input[0] into shape stated by input[1]
    '''
    def __init__(self, inputs: list[str], outputs: list[str]):
        super().__init__(inputs, outputs)

    def forward(self,inputs):
        return inputs[0].reshape(inputs[1])
    
    @classmethod
    def from_onnx_node(self, onnx_model, onnx_node):
        if onnx_node.op_type != 'Reshape':
            raise TypeError('Input node is not a Reshape node')
        
        inputs = onnx_node.input
        outputs = onnx_node.output

        return reshape_node(inputs,outputs)

class shape_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str]):
        '''
        Creates a node that outputs the shape of the input
        '''
        super(shape_node,self).__init__(inputs,outputs)

    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        if onnx_node.op_type != 'Shape':
            raise TypeError('Input node is not a shape node')

        inputs = [onnx_node.input[0]]
        outputs = onnx_node.output

        return shape_node(inputs,outputs)
    
    def forward(self,inputs:np.array):
        a = inputs[0]
        return a.shape
    
class gather_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str], axis:int, indices:np.array):
        '''
        Creates a node that takes a set of index slices of the input
        '''
        super(gather_node,self).__init__(inputs,outputs)
        self.axis = axis
        self.indices = indices

    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        if onnx_node.op_type != 'Gather':
            raise TypeError('Input node is not a gather node')

        inputs = [onnx_node.input[0]]
        outputs = onnx_node.output

        axis = get_attribute_by_name('axis',onnx_node.attribute).i
        indices = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[1]))

        return gather_node(inputs,outputs,axis,indices)
    
    def forward(self,input:np.array):
        a = input[0]
        return a[self.indices]
    
class unsqueeze_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str], axis):
        '''
        Creates a node that unsqueezes in a specific attribute axis
        '''
        super(unsqueeze_node,self).__init__(inputs,outputs)
        self.axis = axis

    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        if onnx_node.op_type != 'Unsqueeze':
            raise TypeError('Input node is not an Unsqueeze node')

        inputs = [onnx_node.input[0]]
        outputs = onnx_node.output

        axis = get_attribute_by_name('axes',onnx_node.attribute).i

        return unsqueeze_node(inputs,outputs,axis)
    
    def forward(self,input:np.array):
        a = input[0]
        a = np.expand_dims(a,axis=self.axis)
        return a

class squeeze_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str]):
        '''
        Creates a node that squeezes in a specific attribute axis
        '''
        super(squeeze_node,self).__init__(inputs,outputs)

    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        if onnx_node.op_type != 'Squeeze':
            raise TypeError('Input node is not an Unsqueeze node')

        inputs = [onnx_node.input[0]]
        outputs = onnx_node.output

        # axis = get_attribute_by_name('axes',onnx_node.attribute).i

        return squeeze_node(inputs,outputs)
    
    def forward(self,input:np.array):
        a = input[0]
        return a.squeeze()


class concat_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str], axis:int, to_concat:np.array):
        '''
        Creates a node that concatenates the input with a predefined initializer
        '''
        super(concat_node,self).__init__(inputs,outputs)
        self.axis = axis
        self.to_concat = to_concat

    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        if onnx_node.op_type != 'Concat':
            raise TypeError('Input node is not a concat node')

        inputs = [onnx_node.input[0]]
        outputs = onnx_node.output

        to_concat = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[1]))

        axis = get_attribute_by_name('axis',onnx_node.attribute).i

        return concat_node(inputs,outputs,axis,to_concat)
    
    def forward(self,inputs:np.array):
        a = inputs[0]
        return np.concatenate([a,self.to_concat],axis=self.axis)

class dequantize_node(Node):
    '''
    Dequantizes the input
    '''

    def __init__(self, inputs:list[str], outputs:list[str], scale:float, zp:float):
        super(dequantize_node,self).__init__(inputs,outputs)
        self.scale = scale
        self.zp = zp

    def forward(self,input:np.array):
        input = np.array(input).squeeze(axis=0)
        return self.scale*(input - self.zp)
    
    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        if onnx_node.op_type != 'DequantizeLinear':
            raise TypeError('Input node is not a dequantize node')

        inputs = [onnx_node.input[0]]
        outputs = onnx_node.output

        scale = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[1]))
        zp = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[2]))

        return dequantize_node(inputs,outputs,scale,zp)

class quantize_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str], scale:float, zp:float):
        '''
        Creates a quantize node that quantizes the inputs
        '''
        super(quantize_node,self).__init__(inputs,outputs)
        self.scale = scale
        self.zp = zp

    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        if onnx_node.op_type != 'QuantizeLinear':
            raise TypeError('Input node is not a quantize node')

        inputs = [onnx_node.input[0]]
        outputs = onnx_node.output

        scale = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[1]))
        zp = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[2]))
        # bitwidth = get_initializer_by_name('bit_width',onnx_node.attribute)

        return quantize_node(inputs,outputs,scale,zp)
    
    def forward(self,input:np.array):
        input = np.array(input).squeeze(axis=0)
        out = np.round((input/self.scale) + self.zp)
        return out

def from_QLinearMatMul(onnx_model,onnx_node):
    '''
    Creates a set of nodes equivalent to an ONNX QLinearMatMul
    
    QLinearConv input structure:
    0. input
    1. input_scale
    2. input_zero_point
    3. kernel
    4. kernel_scale
    5. kernel_zero_point
    6. output_scale
    7. output_zero_point
    '''
    if onnx_node.op_type != 'QLinearMatMul':
        raise TypeError('Input node is not a QLinearMatmul node')

    inputs = [onnx_node.input[0]]
    scaler_input = [onnx_node.input[0]+'_scaler_input']
    outputs = onnx_node.output

    # Quantization-related parameters
    scale_x = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[1]))
    scale_w = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[4]))
    scale_y = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[6]))
    zp_x = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[2]))
    zp_w = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[5]))
    zp_y = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[7]))

    # Matrix Parameters
    matrix = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[3]))
    biases = np.zeros(matrix.shape[1])

    # == M Scaling ==
    # See "tflite quantized matmul" in quantization notes
    scale = (scale_x * scale_w) / scale_y
    
    # == Zero point offset ==
    # See "zeroes thereof" in quantization notes
    # zp_w is always assumed to be 0, otherwise scaling actually gets expensive
    assert zp_w.any() == False
    scaler_offset = zp_y - scale*(matrix.sum(axis=0) * zp_x)
    # scaler_offset = np.array(0) 

    output_nodes = [
        gemm_node(inputs,scaler_input,matrix,biases),
        output_scale_node(scaler_input,outputs,scale=scale,offset=scaler_offset)
    ]

    return output_nodes

def from_QLinearConv(onnx_model,onnx_node,channel_minor=False):
    '''
    Creates a set of nodes equivalent to an ONNX QLinearConv
    
    QLinearConv input structure:
    0. input
    1. input_scale
    2. input_zero_point
    3. kernel
    4. kernel_scale
    5. kernel_zero_point
    6. output_scale
    7. output_zero_point
    8. bias
    '''
    if onnx_node.op_type != 'QLinearConv':
        raise TypeError('Input node is not a QLinearConv node')

    inputs = [onnx_node.input[0]]
    scaler_input = [onnx_node.input[0]+'_scaler_input']
    outputs = onnx_node.output

    # Quantization-related parameters
    scale_x = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[1])).astype(np.float32)
    scale_w = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[4])).astype(np.float32)
    scale_y = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[6])).astype(np.float32)
    zp_x = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[2])).astype(np.int32)
    zp_w = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[5])).astype(np.int32)
    zp_y = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[7])).astype(np.int32)

    # Matrix Parameters
    kernel = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[3])).astype(np.int32)
    biases = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[8])).astype(np.int32)

    strides = get_attribute_by_name('strides',onnx_node.attribute).ints[0]
    group = get_attribute_by_name('group',onnx_node.attribute).i

    # == M Scaling ==
    # See "tflite quantized matmul" in quantization notes
    scale = (scale_x * scale_w) / scale_y
    
    # == Zero point offset ==
    # See "zeroes thereof" in quantization notes
    # zp_w is always assumed to be 0, otherwise scaling actually gets expensive
    assert zp_w.any() == False

    # K C H W kernel

    wadds = zp_x * kernel.sum(axis=(1,2,3)) # ZX*SUM(W) 
    # gadds is always 0 if zp_w is 0
    # gadds = kernel.shape[-1]*kernel.shape[-2]*zp_x*zp_w # general adds, as in N*Z1*Z2

    biases = biases - wadds

    scaler_offset = zp_y #+ scale * (-wadds)
    # scaler_offset = np.array(0)
 
    if group == 1:
        # Regular convolution
        output_nodes = [
            conv_node(inputs,scaler_input,kernel,biases,strides=strides,channel_minor=channel_minor,zero_point=zp_x),
            output_scale_node(scaler_input,outputs,scale,offset=scaler_offset)
        ]
    else:
        # Depthwise convolution
        nodes, concatenator = generate_depthwise_nodes(inputs,scaler_input,kernel,biases,strides, zero_point=zp_x)
        output_nodes = nodes + [concatenator,output_scale_node(scaler_input,outputs,scale=scale,offset=scaler_offset)]

    return output_nodes

def generate_depthwise_nodes(inputs,outputs,kernels,biases,strides,zero_point=0):

    dwnode_outputs = [f'{outputs[0]}_dwch_{i}' for i,ker in enumerate(kernels)]

    nodes = []
    for i,kern in enumerate(kernels):
        output_for_this_dwnode = [f'{outputs[0]}_dwch_{i}']
        nodes.append(conv_node(inputs,output_for_this_dwnode,kernels[i],biases[i],in_channel=i,strides=strides,zero_point=zero_point))

    concatenator = cat_node(dwnode_outputs,outputs) #Make a concatenator node

    return nodes, concatenator

class quantized_global_avg_pool_node(Node):
    '''
    Quantized Avg Pool

    Not hardwarelike
    '''

    def __init__(self, inputs:list[str], outputs:list[str], in_scale:float, in_zp:float, out_scale:float, out_zp:float):
        super(quantized_global_avg_pool_node,self).__init__(inputs,outputs)
        self.in_scale = in_scale
        self.in_zp = in_zp
        self.out_scale = out_scale
        self.out_zp = out_zp

    def forward(self,inputs:np.array):
        a = inputs[0].squeeze(axis=0)

        input_shaped = a.reshape(a.shape[0],-1)
        
        q_xi_sum = np.sum(input_shaped,axis=1)

        n = input_shaped.shape[1]

        m = self.in_scale/(self.out_scale)
        b = - m*self.in_zp + self.out_zp
        
        q_y = (m/n) * q_xi_sum + b

        return q_y
    
    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):

        inputs = [onnx_node.input[0]]
        outputs = onnx_node.output

        in_scale = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[1]))
        in_zp = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[2]))
        out_scale = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[3]))
        out_zp = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[4]))

        return quantized_global_avg_pool_node(inputs,outputs,in_scale,in_zp,out_scale,out_zp)

class quantized_linear_add_node(Node):
    '''
    Implementation of quantized linear add. See
    https://openaccess.thecvf.com/content_cvpr_2018/Supplemental/0777-supp.pdf

    Not hardwarelike -- does not clip precision of the real fixed point multiplier values
    '''

    def __init__(self, inputs:list[str], outputs:list[str], scale_x:float, scale_y:float, zp_x:float, zp_y:float,
                 scale_out:float, zp_out:float, other_initializer = None):
        super(quantized_linear_add_node,self).__init__(inputs,outputs)
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.zp_x = zp_x
        self.zp_y = zp_y
        self.scale_out = scale_out
        self.zp_out = zp_out
        self.other_initializer = other_initializer

    def forward(self,inputs):
        x = inputs[0]
        if self.other_initializer is not None:
            y = self.other_initializer
        else:
            y = inputs[1]
        res_real = (x - self.zp_x) * self.scale_x + (y - self.zp_y) * self.scale_y
        res_q = np.round(res_real / self.scale_out) + self.zp_out
        return res_q
    
    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):

        inputs = [onnx_node.input[0],onnx_node.input[3]]
        other_initializer = None
        
        # Check if either input is an initializer!
        if( is_initializer(onnx_model,onnx_node.input[3]) ):
            inputs = [onnx_node.input[0]]
            other_initializer = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[3]))
        if( is_initializer(onnx_model,onnx_node.input[0]) ):
            inputs = [onnx_node.input[3]]
            other_initializer = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[0]))
        
        outputs = onnx_node.output

        scale_x = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[1]))
        zp_x = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[2]))
        scale_y = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[4]))
        zp_y = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[5]))
        scale_out = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[6]))
        zp_out = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[7]))

        return quantized_linear_add_node(inputs,outputs,scale_x,scale_y,zp_x,zp_y,scale_out,zp_out,other_initializer)

class output_scale_node(Node):
    def __init__(self, inputs, outputs, scale, offset, scale_precision = 32, out_precision=8):
        '''
        Output scaling per TFLite quantization

        all inputs have scale, but only x can have a zero

        Hardwarelike -- Clips precision of the real_scale to scale_precision
        '''
        super(output_scale_node,self).__init__(inputs,outputs)
        
        self.real_scale = scale

        m0, shift = q.vconvert_scale_to_shift_and_m0(
                        scale,
                        precision=scale_precision
        )

        fp_m = m0 * np.power(2.,shift)

        self.m0 = m0
        self.shift = shift
        self.fp_m = fp_m
        self.offset = offset

        self.out_precision = out_precision
        self.scale_precision = scale_precision

    def forward(self,input:np.array):
        input_squeezed = np.array(input).squeeze()
        # black magic needed to force fp_m multiplication to broadcast along the first dimension of out (C)

        fp_m = self.fp_m
        fp_m = fp_m.reshape(fp_m.shape[0],*([1]*(len(input_squeezed.shape)-1)))
        out = input_squeezed * fp_m
        if self.offset.ndim == 0:
            offset_broadcast = self.offset
        else:
            offset_broadcast = self.offset[:,*[np.newaxis]*len(input_squeezed.shape[1:])]
        out = out + offset_broadcast
        out = out.round()

        out = q.saturating_clip(out, self.out_precision, signed=False)
        out = out.reshape(1,*out.shape)
        return out

class conv_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str], kernel:np.array, biases:np.array, in_channel = None, strides = 1, channel_minor = False, zero_point = 0):
        '''
        Creates a conv node from a kernel of shape K,C,H,W

        Matrix version is channel major (K, C H W) by default
        If channel_minor is True, then it is row major (K, H W C)

        Zero-point is used to zero-pad integer-only convolutions.
        In integer-only convolutions, the zero-point is the value needed to be used for 
        zero padding, as it's the value that represents zero.
        '''
        super(conv_node,self).__init__(inputs,outputs)
        self.kernel = kernel
        self.biases = biases

        if channel_minor:
            self.matrix = kernel.transpose(0,2,3,1).reshape(kernel.shape[0],-1).T
        else:
            self.matrix = kernel.reshape(kernel.shape[0],-1).T
        
        self.in_channel = in_channel #for depthwise convolutions
        if in_channel is not None:
            self.depthwise = True
        else:
            self.depthwise = False
        self.strides = strides
        self.channel_minor = channel_minor

        # For integer-only convolutions
        self.zero_point = zero_point

    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        if onnx_node.op_type != 'Conv':
            raise TypeError('Input node is not a convolution')
        if get_attribute_by_name('group',onnx_node.attribute).i > 1:
            raise TypeError('Input convolution is depthwise, use <from_onnx_depthwise> instead')

        inputs = [onnx_node.input[0]]
        outputs = onnx_node.output

        strides = get_attribute_by_name('strides',onnx_node.attribute).ints[0]
        
        kernel = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[1]))
        biases = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[2]))

        return conv_node(inputs,outputs,kernel,biases,strides=strides)
    
    @classmethod
    def from_onnx_depthwise(self,onnx_model,onnx_node):        
        if onnx_node.op_type != 'Conv':
            raise TypeError('Input node is not a convolution')
        if get_attribute_by_name('group',onnx_node.attribute).i == 1:
            raise TypeError('Input convolution is not depthwise, use <from_onnx_node> instead')

        inputs = [onnx_node.input[0]]
        outputs = onnx_node.output

        strides = get_attribute_by_name('strides',onnx_node.attribute).ints[0]

        kernels = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[1]))
        biases = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[2]))

        dwnode_outputs = [f'{outputs[0]}_dwch_{i}' for i,ker in enumerate(kernels)]

        nodes = []
        for i,kern in enumerate(kernels):
            output_for_this_dwnode = [f'{outputs[0]}_dwch_{i}']
            nodes.append(conv_node(inputs,output_for_this_dwnode,kernels[i],biases[i],in_channel=i,strides=strides))

        concatenator = cat_node(dwnode_outputs,outputs) #Make a concatenator node

        return nodes, concatenator
    
    def forward(self,input:np.array):
        '''
        Performs the node operation

        input[0]: C,H,W tensor
        '''
        # Squeeze to remove batch dimension
        input = np.array(input[0]).squeeze(axis=0)

        if self.in_channel is not None:
            input = np.expand_dims(input[self.in_channel],0)

        flat_input = toeplitzize_input(input,ksize=self.kernel.shape[-1],strides=self.strides,channel_minor=self.channel_minor, zero_point=self.zero_point)
        # print(flat_input.shape)

        flat_out = flat_input @ self.matrix
        for row in flat_out:
            row+=self.biases
            
        # output a C,H,W tensor
        H = input.shape[1] // self.strides
        C = flat_out.shape[1] 
        W = input.shape[2] // self.strides

        out = flat_out.T.reshape(1,C,H,W)
        if self.in_channel is not None:
            out = out.squeeze(axis=0)
        # out = out.reshape(1,*out.shape)

        return out
    
class clip_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str], range:tuple[float,float]):
        '''
        Creates a clipping node that clips between the specified range
        '''
        super(clip_node,self).__init__(inputs,outputs)
        self.range = range

    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        if onnx_node.op_type != 'Clip':
            raise TypeError('Input node is not a convolution')
        
        inputs = [onnx_node.input[0]]
        outputs = onnx_node.output

        const_node_min = get_node_by_output(onnx_model,onnx_node.input[1])
        const_node_max = get_node_by_output(onnx_model,onnx_node.input[2])

        min = nphelp.to_array(const_node_min.attribute[0].t)
        max = nphelp.to_array(const_node_max.attribute[0].t)
        range = (min,max)

        return clip_node(inputs,outputs,range)
    
    def forward(self,input:np.array):
        input = np.array(input).squeeze(axis=0)
        return np.clip(input,self.range[0],self.range[1])
    
class add_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str]):
        '''
        Creates an add node that adds the inputs
        '''
        super(add_node,self).__init__(inputs,outputs)

    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        if onnx_node.op_type != 'Add':
            raise TypeError('Input node is not an add node')

        inputs = onnx_node.input
        outputs = onnx_node.output

        return add_node(inputs,outputs)
    
    def forward(self,inputs:list[np.ndarray]):
        inputs = np.array(inputs).squeeze()
        return np.sum(inputs,axis=0)

class cat_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str],axis:int = None):
        '''
        Creates an add node that concatenates the inputs
        in the order that they appear in the input list
        '''
        super(cat_node,self).__init__(inputs,outputs)
        self.axis = axis
    
    def forward(self,inputs:np.array):
        '''
        Calls np.concatenate on axis=self.axis
        Returns the input if the concat axis is not in the range or self.axis is unspecified.
        '''
        if self.axis is None:
            inputs = np.array(inputs)
            return inputs
        return np.concatenate(inputs,axis=self.axis)
    
class global_avg_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str]):
        '''
        Creates an averaging node that averages every channel,
        leaving a [C,1,1,...] array.
        '''
        super(global_avg_node,self).__init__(inputs,outputs)

    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        if onnx_node.op_type != 'GlobalAveragePool':
            raise TypeError('Input node is not a GlobalAveragePool node')

        inputs = onnx_node.input
        outputs = onnx_node.output

        return global_avg_node(inputs,outputs)
    
    def forward(self,inputs:np.array):
        inputs = np.array(inputs).squeeze(axis=0)
        return np.average(inputs,axis=(1,2))
    
class flatten_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str]):
        '''
        Creates an flattening node that flattens the inputs
        '''
        super(flatten_node,self).__init__(inputs,outputs)

    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        if onnx_node.op_type != 'Flatten':
            raise TypeError('Input node is not a Flatten node')

        inputs = onnx_node.input
        outputs = onnx_node.output

        return flatten_node(inputs,outputs)
    
    def forward(self,inputs:np.array):        
        inputs = np.array(inputs).squeeze(axis=0)
        return inputs.flatten()

class gemm_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str], matrix:np.array, biases:np.array = 0):
        '''
        Creates a general matrix multiply node from a matrix of shape H,W
        '''
        super(gemm_node,self).__init__(inputs,outputs)
        self.matrix = matrix
        self.biases = biases

    @classmethod
    def from_onnx_node(self,onnx_model,onnx_node):
        if onnx_node.op_type != 'Gemm':
            raise TypeError('Input node is not a gemm')
        
        inputs = [onnx_node.input[0]]
        outputs = onnx_node.output

        matrix = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[1])).T
        biases = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[2]))

        return gemm_node(inputs,outputs,matrix,biases)
        
    def forward(self,input:np.array):
        '''
        Performs the node operation

        input: vector
        '''
        input = np.array(input).squeeze(axis=0)
        out = (input @ self.matrix) + self.biases

        return out
    
class toeplitzizer_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str], ksize:int, strides:int = 1):
        '''
        Creates a node that transforms the input into a flattened toeplitz
        matrix at the output
        '''
        super(toeplitzizer_node,self).__init__(inputs,outputs)
        self.ksize = ksize
        self.strides = strides
        
    def forward(self,input:np.array):
        '''
        Performs the node operation

        input: vector
        '''
        input = np.array(input[0]).squeeze(axis=0)
        out = toeplitzize_input(input,ksize=self.ksize,strides=self.strides)
        return out
    
class slicer_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str], row_lim:list[int] = [None,None],col_lim:list[int] = [None,None]):
        '''
        Creates a node that slices the input array according to the slicing
        defined by [row_lim[0]:row_lim[1],col_lim[0]:col_lim[1]]
        '''
        super(slicer_node,self).__init__(inputs,outputs)
        self.row_lim = row_lim
        self.col_lim = col_lim

    def forward(self,input:np.array):
        input = np.array(input).squeeze(axis=0)
        if len(input.shape) == 1:
            lim = self.col_lim if self.row_lim == [None,None] else self.row_lim
            return input[lim[0]:lim[1]]
        return input[self.row_lim[0]:self.row_lim[1],self.col_lim[0]:self.col_lim[1]]
    
class reshaper_node(Node):
    def __init__(self, inputs:list[str], outputs:list[str], channels:int):
        '''
        Creates a node that reshapes the input array into C x H x W where H = W
        '''
        super(reshaper_node,self).__init__(inputs,outputs)
        self.channels = channels

    def forward(self,input:np.array):
        input = np.array(input).squeeze(axis=0)
        C = self.channels
        H = int(np.sqrt(input.shape[0]))
        W = H
        return input.T.reshape(C,H,W)