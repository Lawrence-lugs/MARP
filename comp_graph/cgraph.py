


import onnx
from onnx import numpy_helper as nphelp
import numpy as np

def get_recfield_for_pixel(r,c,matrix,ksize):
    'Obtains the receptive field for a convolution output pixel'
    if ksize == 1:
        return matrix[r,c]
    if ksize == 3:
        return matrix[r:r+3,c:c+3].transpose(2,0,1)

def toeplitzize_input(in_tensor,ksize=3,strides=1):
    '''
    * Flattens input tensor into a Toeplitz matrix for passing into a
    flattened kernel.
    * Zero pads by default.

    input: C,H,W tensor
    '''

    #Convert to H,W,C tensor
    tensor = in_tensor.transpose(1,2,0)

    H = tensor.shape[0] // strides
    W = tensor.shape[1] // strides
    C = tensor.shape[2]

    if ksize == 3:
        tensor2 = np.pad(tensor,((1,1),(1,1),(0,0)))
        out = np.empty((H*W,C*9))
    else:
        tensor2 = tensor
        out = np.empty((H*W,C))

    for r in range(H):
        for c in range(W):
            out[r*W + c] = get_recfield_for_pixel(strides*r,strides*c,tensor2,ksize).flatten()

    return out

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

class node(object):
    def __init__(self, inputs:list[str], outputs:list[str]):
        self.inputs = inputs 
        self.outputs = outputs

    def forward(self):
        raise NotImplementedError

class conv_node(node):
    def __init__(self, inputs:list[str], outputs:list[str], kernel:np.array, biases:np.array, in_channel = None, strides = 1):
        '''
        Creates a conv node from a kernel of shape O,I,H,W
        '''
        super(conv_node,self).__init__(inputs,outputs)
        self.kernel = kernel
        self.biases = biases
        self.matrix = kernel.reshape(kernel.shape[0],-1)
        self.in_channel = in_channel
        self.strides = strides

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

        input: C,H,W tensor
        '''
        if self.in_channel is not None:
            input = np.expand_dims(input[self.in_channel],0)

        flat_input = toeplitzize_input(input,ksize=self.kernel.shape[-1],strides=self.strides)
        # print(flat_input.shape)

        flat_out = self.matrix @ flat_input.T
        for col in flat_out.T:
            col+=self.biases
            
        # output a C,H,W tensor
        C = flat_out.shape[0] 
        H = input.shape[1] // self.strides
        W = input.shape[2] // self.strides

        out = flat_out.reshape(C,H,W)
        if self.in_channel is not None:
            out = out.squeeze()

        return out
    
class clip_node(node):
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
        return np.clip(input,self.range[0],self.range[1])
    
class add_node(node):
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
    
    def forward(self,inputs:np.array):
        return np.sum(inputs,axis=0)

class cat_node(node):
    def __init__(self, inputs:list[str], outputs:list[str]):
        '''
        Creates an add node that concatenates the inputs
        in the order that they appear in the input list
        '''
        super(cat_node,self).__init__(inputs,outputs)
    
    def forward(self,inputs:np.array):
        ' This is a formality, the inputs will already be stacked on each other '
        return inputs
    
class global_avg_node(node):
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
        return np.average(inputs,axis=(1,2))
    
class flatten_node(node):
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
        return inputs.flatten()

class gemm_node(node):
    def __init__(self, inputs:list[str], outputs:list[str], matrix:np.array, biases:np.array):
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

        matrix = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[1]))
        biases = nphelp.to_array(get_initializer_by_name(onnx_model,onnx_node.input[2]))

        return gemm_node(inputs,outputs,matrix,biases)
        
    def forward(self,input:np.array):
        '''
        Performs the node operation

        input: vector
        '''
        out = (self.matrix @ input) + self.biases

        return out