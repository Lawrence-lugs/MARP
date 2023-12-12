import onnx
from onnx import numpy_helper as nphelp
import numpy as np
import cnodes
from tqdm import tqdm
import splitter
import rectpack

class cgraph(object):
    '''
    A computational graph

    Attributes
    ----------
    nodes : list[cnodes.node]
        A list of computational nodes which perform functions on
        the arrays from its input edges as passes 
    edges : dict[str,np.ndarray]
        A dictionary of tensors which serve as inputs and outputs
        of nodes on the cgraph.
    '''

    def __init__(self,node_list:list[cnodes.node]):
        '''
        Parameters
        ---------
        nodes : list[cnodes.node]
            List of computational nodes from which to build the cgraph 
        '''

        self.nodes = node_list
        
        self.edges = dict()
        for node in self.nodes:
            for input in node.inputs:
                self.edges[input] = None
            for output in node.outputs:
                self.edges[output] = None

    @classmethod
    def from_onnx_model(cls,nx_model, tiles=None):
        '''
        Obtains a `cgraph` from an ONNX model loaded in.
        '''
        node_list = []
        for node in nx_model.graph.node:
            if node.op_type == 'Conv':
                if node.attribute[1].i == 1:
                    node_list.append(cnodes.conv_node.from_onnx_node(nx_model,node))
                else:
                    convs,catter = cnodes.conv_node.from_onnx_depthwise(nx_model,node)
                    node_list.extend(convs)
                    node_list.append(catter)    
            if node.op_type == 'Gemm':
                node_list.append(cnodes.gemm_node.from_onnx_node(nx_model,node))
            if node.op_type == 'Add':
                node_list.append(cnodes.add_node.from_onnx_node(nx_model,node))
            if node.op_type == 'Clip':
                node_list.append(cnodes.clip_node.from_onnx_node(nx_model,node))
            if node.op_type == 'Flatten':
                node_list.append(cnodes.flatten_node.from_onnx_node(nx_model,node))
            if node.op_type == 'GlobalAveragePool':
                node_list.append(cnodes.global_avg_node.from_onnx_node(nx_model,node))

        return cgraph(node_list)

    def forward(self,input_dict:dict,output_keys:list[str] = None,verbose=False):
        '''
        Parameters
        ----------
        input_dict : dictionary
            dictionary that contains keys corresponding to keys
            in the edges dictionary, and values corresponding to numpy
            array inputs to place in those edges.
        output_keys : list[str], optional
            list containing the keys of the edges whose values will
            be returned after inference.
        '''
        def check_if_node_ready(node):
            for input in node.inputs:
                if self.edges[input] is None:
                    return False
            return True
        
        for key in input_dict:
            self.edges[key] = input_dict[key]

        for node in tqdm(self.nodes,disable=not verbose):
            check_if_node_ready(node)
            if verbose: print(f'node:{node},\n output: {node.outputs}')
            in_array = []
            for input in node.inputs:
                in_array.append(self.edges[input])
            
            # Squeeze eliminates extra dimension if there's only one input
            # This causes ragged nested arrays sometimes. 

            out_array = node.forward(in_array)
            for output in node.outputs:
                self.edges[output] = out_array 

        if output_keys is None:
            # Return the output of the last node
            output_keys = self.nodes[-1].outputs

        return [self.edges[x] for x in output_keys]
    
    def _get_shape_list_id(self):
        '''
        Obtains the list of shapes of all matrix-containing nodes
        and assigns an ID to each depending on their position in the node list
        '''
        shapes = []
        for id,node in enumerate(self.nodes):
            if hasattr(node,"matrix"):
                shapes.append( (*node.matrix.shape,id) )
        return shapes

def split_convolutions(in_cgraph:cgraph,H:int,W:int):

    new_nodes = []

    for node in in_cgraph.nodes:
        if type(node) == cnodes.conv_node:
            replacements = splitter.split_conv_into_chunks(node,H,W)
            new_nodes.extend(replacements)
        elif type(node) == cnodes.gemm_node:
            replacements = splitter.split_gemm_into_chunks(node,H,W)
            new_nodes.extend(replacements)
        else:
            new_nodes.append(node)

    return cgraph(new_nodes)