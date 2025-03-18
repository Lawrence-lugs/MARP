import onnx
from onnx import numpy_helper as nphelp
import numpy as np
from . import cnodes
from tqdm import tqdm
from . import splitter
import os

class Cgraph(object):
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

    def __init__(self,node_list:list[cnodes.Node]): 
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

    def check_if_node_done(self,node):
        '''
        Checks if the outputs of a node is already filled
        '''
        for output in node.outputs:
            if self.edges[output] is None:
                return False
        return True

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
            elif node.op_type == 'Gemm':
                node_list.append(cnodes.gemm_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'Add':
                node_list.append(cnodes.add_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'Clip':
                node_list.append(cnodes.clip_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'Flatten':
                node_list.append(cnodes.flatten_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'GlobalAveragePool':
                node_list.append(cnodes.global_avg_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'QuantizeLinear':
                node_list.append(cnodes.quantize_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'QLinearConv':
                node_list.extend(cnodes.from_QLinearConv(nx_model,node))
            elif node.op_type == 'QLinearAdd':
                node_list.append(cnodes.quantized_linear_add_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'QLinearGlobalAveragePool':
                node_list.append(cnodes.quantized_global_avg_pool_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'DequantizeLinear':
                node_list.append(cnodes.dequantize_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'Shape':
                node_list.append(cnodes.shape_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'Reshape':
                node_list.append(cnodes.reshape_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'Gather':
                node_list.append(cnodes.gather_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'Unsqueeze':
                node_list.append(cnodes.unsqueeze_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'Concat':
                node_list.append(cnodes.concat_node.from_onnx_node(nx_model,node))
            elif node.op_type == 'QLinearMatMul':
                node_list.extend(cnodes.from_QLinearMatMul(nx_model,node))
            else:
                raise NotImplementedError(f'Node type {node.op_type} not implemented')

        return Cgraph(node_list)

    def check_if_node_ready(self,node):
        '''
        Checks if the preceding edges of a node is already filled
        '''
        for input in node.inputs:
            if self.edges[input] is None:
                return False
        return True

    def forward(self,input_dict:dict,output_keys:list[str] = None,verbose=False,cachePath = None, recalculate=False, progbar=True):
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

        if cachePath is not None:
            if not os.path.exists(cachePath):
                os.makedirs(cachePath)
        
        for key in input_dict:
            self.edges[key] = input_dict[key]

        while not self.check_if_node_done(self.nodes[-1]):
            for node in tqdm(self.nodes,disable=not progbar):
                if not self.check_if_node_ready(node):
                    continue

                if self.check_if_node_done(node):
                    continue

                if verbose: print(f'node:{node},\n output: {node.outputs}')
                in_array = []

                for input in node.inputs:
                    in_array.append(self.edges[input])

                if cachePath is not None:
                    if recalculate:
                        out_array = node.forward(in_array)
                        np.save(cachePath + f'/{node.outputs[0]}.npy',out_array)
                    else:
                        try:
                            out_array = np.load(cachePath + f'/{node.outputs[0]}.npy',allow_pickle=True)
                        except FileNotFoundError:
                            out_array = node.forward(in_array)
                            np.save(cachePath + f'/{node.outputs[0]}.npy',out_array) 
                else:
                    out_array = node.forward(in_array)

                for output in node.outputs:
                    self.edges[output] = out_array

            if output_keys is None:
                # Return the output of the last node
                output_keys = self.nodes[-1].outputs

            return [self.edges[x] for x in output_keys]
    
    def _get_shape_list_id(self, excludeDepthwise=True):
        '''
        Obtains the list of shapes of all matrix-containing nodes
        and assigns an ID to each depending on their position in the node list
        '''
        shapes = []
        for id,node in enumerate(self.nodes):
            if hasattr(node,"matrix"):
                if excludeDepthwise:
                    if '_dwch_' in node.outputs[0]:
                        continue    
                shapes.append( (*node.matrix.shape,id) )
            node.rid = id
        return shapes

    def color_by_separable_type(self):
        '''
        Colors pointwise, depthwise, gemms, and conv nodes.

        red = gemm
        blue = pointwise conv
        green = regular conv

        Depthwise convolutions are not included in the coloring.
        '''

        for node in self.nodes:
            if hasattr(node,'from_type'):
                if node.from_type == 'gemm':
                    node.color = '#f1948a' # Red
                elif node.from_type == 'pointwise':
                    node.color = '#85c1e9' # Blue
                else:
                    node.color = '#58d68d' # Green

        return self.generate_color_dict()
    
    def color_by_nid(self):

        highnumber = 12312941249
        for node in self.nodes:
            rand = highnumber*(node.nid)
            node.color = f'#{hex(rand)[2:8]}'
        return self.generate_color_dict()

    def generate_color_dict(self):
        '''
        Generates a color dictionary for the nodes in the cgraph
        '''
        color_dict = dict()
        for node in self.nodes:
            if hasattr(node,'color') and hasattr(node,'rid'):
                color_dict[node.rid] = node.color
        return color_dict
    
    def rid_to_nid_dict(self):
        '''
        Generates a dictionary that maps rid to nid
        '''
        rid_to_nid = dict()
        for node in self.nodes:
            if hasattr(node,'rid'):
                rid_to_nid[node.rid] = node.nid
        return rid_to_nid

def split_convolutions(in_cgraph:Cgraph,H:int,W:int):
    '''
    Limits the cgraph's matrices to a size limit of H, W.
    If the matrix's size is larger, the matrix is split and 
    additional nodes to manage this computation is added.
    '''
    new_nodes = []

    for nid,node in enumerate(in_cgraph.nodes):
        if type(node) == cnodes.conv_node:
            replacements = splitter.split_conv_into_chunks(node,H,W)
            for node in replacements:
                node.nid = nid
            new_nodes.extend(replacements)
        elif type(node) == cnodes.gemm_node:
            replacements = splitter.split_gemm_into_chunks(node,H,W)
            for node in replacements:
                node.nid = nid
            new_nodes.extend(replacements)
        else:
            node.nid = nid
            new_nodes.append(node)

    return Cgraph(new_nodes)

