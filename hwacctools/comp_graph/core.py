from . import splitter,cnodes,cgraph, packer_utils  as pu
from ..onnx_tools import onnx_splitter
import numpy as np
import onnx
import matplotlib.pyplot as plt
import rectpack
import seaborn as sns
from ..quantization import quant as q

def get_ids_for_shapelist(shapelist):
    outlist = []
    for i,shape in enumerate(shapelist):
        outlist.append( (*shape,i) )
    return outlist

def add_rects_to_packer(packer,shapelist):
    for shape in shapelist:
        # Shape in numpy is y,x but we want x,y
        # print(shape)
        rectw = shape[1]
        recth = shape[0]
        rid = shape[2]
        packer.add_rect(rectw,recth,rid)
    return packer

def pack_matrices(cgraph,imc_core_size,packer):
    inshapes = cgraph.split_convolutions(inshapes,H=imc_core_size[0],W=imc_core_size[1])
    cgraph_shapes = inshapes._get_shape_list_id()

    packer = rectpack.newPacker(rotation=False,
                                pack_algo=rectpack.MaxRectsBssf)
    packer = add_rects_to_packer(packer,cgraph_shapes)

    packer.add_bin(*imc_core_size,count=float("inf"))
    packer.pack()
    return cgraph,imc_core_size,packer

def _infer_flattened_matrix_from_kernel(kernel_shape):

    # Kernels are KCHW

    ydim = np.prod(kernel_shape[1:])
    xdim = kernel_shape[0]    

    return xdim, ydim

def _get_aimc_mapped_shapes_from_onnx(nx_model : onnx.ModelProto):
    '''
    Returns a list of matrix shapes from an ONNX model
    and a dictionary of node IDs to name.
    '''
    shapes = []
    nid_to_name = {}
    for node_id,node in enumerate(nx_model.graph.node):
        if node.op_type == 'QLinearConv':
            
            groups_attr = next(i for i in node.attribute if i.name == 'group')
            groups = onnx.helper.get_attribute_value(groups_attr)
            if groups != 1:
                # Skip grouped convolutions (not mappable to AIMC)
                continue

            kernel_name = node.input[3]
            init = next((init for init in nx_model.graph.initializer if init.name == kernel_name), None)
            xdim, ydim = _infer_flattened_matrix_from_kernel(init.dims)
            shapes.append((xdim, ydim, node_id))
            nid_to_name[node_id] = node.name
        elif node.op_type == 'QLinearMatMul':
            weight_name = node.input[3]
            init = next((init for init in nx_model.graph.initializer if init.name == weight_name), None)
            shape = tuple(d for d in init.dims)
            shapes.append((shape[1], shape[0], node_id))
            nid_to_name[node_id] = node.name
    return shapes, nid_to_name

def check_packing_success(packer, shapes):
    '''
    Checks if the packing was successful by comparing the number of packed rectangles
    with the number of shapes.
    '''
    if len(packer.rect_list()) != len(shapes):
        print(f'Packing incomplete: packed {len(packer.rect_list())} vs {len(shapes)} matrices in cgraph')
        return False
    else:
        print(f'Packing successful: packed {len(packer.rect_list())} vs {len(shapes)} matrices in cgraph in {len(packer)} bins')
        return True
    
def pack_shapes_into_coresize_bins(packer, shapes, imc_core_size):

    packer.add_bin(*imc_core_size,count=float("inf"))
    packer = add_rects_to_packer(packer,shapes)
    packer.pack()

    return packer

def get_mapped_matrices_from_packed_cgraph(cgraph, packer, imc_core_size):
    '''
    Takes a cgraph and a packer, and returns a list of matrices of size imc_core_size for each bin of packer.
    '''

    bin_matrices = []
    for bin_id,bin in enumerate(packer):

        # Prepare matrix for a new core
        cell_array = np.zeros(imc_core_size)

        # Populate it with the mapped matrices 
        for mapped_rect in bin:
            mapped_node = cgraph.nodes[mapped_rect.rid]

            # Attach mapping information to the cnode
            mapped_node.bin_id = bin_id
            mapped_node.offset_x = mapped_rect.x
            mapped_node.offset_y = mapped_rect.y

            x1 = mapped_rect.x
            x2 = mapped_rect.corner_top_r.x
            y1 = mapped_rect.y
            y2 = mapped_rect.corner_top_r.y

            cell_array[y1:y2,x1:x2] = mapped_node.matrix

        bin_matrices.append(cell_array)

    return bin_matrices

def _get_kernel_of_nx_node(nx_node : onnx.NodeProto, nx_model : onnx.ModelProto) -> np.ndarray:
    '''
    Returns the kernel of an ONNX node as a numpy array.
    Assumes the node is a QLinearConv or QLinearMatMul.
    '''
    if nx_node.op_type not in ['QLinearConv', 'QLinearMatMul']:
        raise ValueError(f"Node {nx_node.name} is not a QLinearConv or QLinearMatMul")
    
    kernel = get_init_of_nx_node(nx_node, nx_model, 3) 
    if kernel.ndim == 2:
    # If the kernel is 2D, it is a QLinearMatMul
        kernel = kernel.reshape((kernel.shape[1],kernel.shape[0], 1, 1)) # Pretend it's a pointwise convolution

    return kernel

def get_init_of_nx_node(nx_node : onnx.NodeProto, nx_model : onnx.ModelProto, input_id) -> np.ndarray:
    '''
    Returns the initializer of an ONNX node as a numpy array.
    '''
    init_name = nx_node.input[input_id]
    init = next((init for init in nx_model.graph.initializer if init.name == init_name), None)
    if init is None:
        raise ValueError(f"Initializer {init_name} not found in ONNX model")
    return onnx.numpy_helper.to_array(init)

def _get_matrix_from_kernel(kernel : np.ndarray, nx_node : onnx.NodeProto) -> np.ndarray:
    '''
    Obtains a channel-minor matrix from a kernel of an ONNX node
    If the ONNX node is a QLinearMatMul, it simply returns the weight matrix
    '''
    return kernel.transpose(0, 2, 3, 1).reshape(kernel.shape[0], -1).T
    
class MappedBin(object):
    '''
    Represents a packed bin of matrices with its ID and offset
    '''
    def __init__(self, bin_id, weights : np.ndarray):
        self.bin_id = bin_id
        self.weights = weights

    def __repr__(self):
        plt.imshow(self.weights, cmap='YlOrBr', vmax=self.weights.max(), vmin=self.weights.min(), origin='lower')
        plt.title(f"Bin {self.bin_id}")
        return f"MappedBin(id={self.bin_id}, shape={self.weights.shape})"

def check_if_depthwise(nx_node : onnx.NodeProto) -> bool:
    '''
    Checks if an ONNX node is a depthwise convolution
    '''
    if nx_node.op_type != 'QLinearConv':
        return False
    groups_attr = next((attr for attr in nx_node.attribute if attr.name == 'group'), None)
    groups = onnx.helper.get_attribute_value(groups_attr) if groups_attr else 1
    return groups > 1

def _get_overall_scale_factors(w_scale, x_scale, y_scale, length):
    '''
    Computes the overall scale factor for a QLinearConv or QLinearMatMul node
    '''
    scale = (w_scale * x_scale) / y_scale  

    if scale.ndim == 0:
        return np.full((length,), scale, dtype=np.float32)
    elif len(scale) == 1:
        return np.full((length,), scale[0], dtype=np.float32)
    elif len(scale) == length:
        return scale.astype(np.float32)

class MappedQRAccNode(object):
    '''
    Represents a QRAcc mapped ONNX QLinearConv or QLinearMatmul node with its matrix shape and ID
    '''
    def __init__(self, nx_node : onnx.NodeProto, bin_id, mapped_rect, nx_model : onnx.ModelProto, offset_x = 0, offset_y = 0):

        self.depthwise = check_if_depthwise(nx_node)

        self.bin_id = bin_id
        self.nx_node = nx_node
         
        # Inefficient, but guaranteed to find the node ID
        self.node_id = next((i for i, n in enumerate(nx_model.graph.node) if n.name == nx_node.name), None)
        if mapped_rect is None:
            self.offset_x = offset_x
            self.offset_y = offset_y
        else:
            self.offset_x = mapped_rect.y
            self.offset_y = mapped_rect.x

        self.kernel = _get_kernel_of_nx_node(nx_node, nx_model).astype(np.int8)  # Ensure kernel is in int8 format
        self.matrix = _get_matrix_from_kernel(self.kernel, nx_node).astype(np.int8)  # Ensure matrix is in int8 format     

        self.name = nx_node.name
        self.type = nx_node.op_type

        pads_attr = next((attr for attr in nx_node.attribute if attr.name == 'pads'), None)
        self.pads = pads_attr.ints if pads_attr else (0, 0, 0, 0)
        strides_attr = next((attr for attr in nx_node.attribute if attr.name == 'strides'), None)
        self.strides = strides_attr.ints if strides_attr else (1, 1)

        self.x_scale = get_init_of_nx_node(nx_node, nx_model, 1)
        self.x_zp = get_init_of_nx_node(nx_node, nx_model, 2)
        self.w_scale = get_init_of_nx_node(nx_node, nx_model, 4)
        self.w_zp = get_init_of_nx_node(nx_node, nx_model, 5)
        self.y_scale = get_init_of_nx_node(nx_node, nx_model, 6)
        self.y_zp = get_init_of_nx_node(nx_node, nx_model, 7)

        self.biases = get_init_of_nx_node(nx_node, nx_model, 8) if len(nx_node.input) > 8 else 0
        ifmap_zp_offset = self.x_zp * self.kernel.sum(axis=(1,2,3)) if self.kernel.ndim == 4 else self.x_zp * self.kernel.sum(axis=0)
        self.biases = self.biases - ifmap_zp_offset # Fold the zero point offset contribution into the overall biases

        # repeat biases to match the number of output channels if biases is a scalar
        if np.isscalar(self.biases):
            self.biases = np.full((self.kernel.shape[0],), self.biases, dtype=np.int32)

        self.scale = _get_overall_scale_factors(
            self.w_scale, 
            self.x_scale, 
            self.y_scale, 
            self.kernel.shape[0]
        ).astype(np.float32)
        
        return

    def __repr__(self):
        return f"MappedQRAccNode(id={self.node_id}, name={self.name}, type={self.type}, bin_id={self.bin_id}, shape={self.matrix.shape}, depthwise={self.depthwise})"

class NxModelMapping(object):
    '''
    Maps an ONNX model to a cgraph and packs it into imc_core_size sized matrices
    Must take in split onnx model already
    '''
    def __init__(self,
                 nx_model : onnx.ModelProto,
                 imc_core_size : tuple[int] = (256, 256),
                 dwc_core_size : int = 32,
                 packer = None,
                 **kwargs
                 ):
        
        onnx_splitter.split_model_to_per_channel(nx_model.graph, C_max = imc_core_size[0], K_max = imc_core_size[1], dwC_max=dwc_core_size)
        nx_shapes, nid_to_name = _get_aimc_mapped_shapes_from_onnx(nx_model)

        if packer is None: packer = rectpack.newPacker(mode=rectpack.PackingMode.Offline, rotation=False, pack_algo=rectpack.MaxRectsBssf)

        packer = pack_shapes_into_coresize_bins(packer, nx_shapes, imc_core_size)
        check_packing_success(packer, nx_shapes)

        self.nid_to_name = nid_to_name
        self.nbins = len(packer)
        self.rects = nx_shapes
        self.packer = packer
        self.core_size = imc_core_size
        self.nx_model = nx_model

        self.mapped_nodes = []
        self.mapped_bins = []
        self._setup_mapping_from_packed_onnx()
        self._setup_mapping_of_depthwise_nodes()

        # Reorder mapped nodes by their node_id
        self.mapped_nodes = sorted(self.mapped_nodes, key=lambda node: node.node_id)

        return
    
    def _setup_mapping_of_depthwise_nodes(self):

        for node_id,node in enumerate(self.nx_model.graph.node):
            if node.op_type == 'QLinearConv':
                groups_attr = next((attr for attr in node.attribute if attr.name == 'group'), None)
                groups = onnx.helper.get_attribute_value(groups_attr) if groups_attr else 1
                if groups > 1:
                    # This is a depthwise convolution, just generate a MappedQRAccNode for it
                    mapped_node = MappedQRAccNode(
                        nx_node = node, 
                        bin_id = None,  # Depthwise nodes are not packed
                        mapped_rect = None,  # No rectangle for depthwise nodes
                        nx_model = self.nx_model
                    )
                    self.mapped_nodes.append(mapped_node)

        return
    
    def _setup_mapping_from_packed_onnx(self):

        for bin_id,bin in enumerate(self.packer):

            # Prepare matrix for a new core
            cell_array = np.zeros(self.core_size, dtype=np.int8)

            # Populate it with the mapped nodes 
            for mapped_rect in bin:
                nx_node = self.nx_model.graph.node[mapped_rect.rid]

                # Attach mapping information to the node
                mapped_node = MappedQRAccNode(
                    nx_node = nx_node, 
                    bin_id = bin_id, 
                    mapped_rect = mapped_rect, 
                    nx_model = self.nx_model
                )
                self.mapped_nodes.append(mapped_node)

                # Fill up the weight matrix of the bin
                x1 = mapped_rect.x
                x2 = mapped_rect.corner_top_r.x
                y1 = mapped_rect.y
                y2 = mapped_rect.corner_top_r.y
                kernel = _get_kernel_of_nx_node(nx_node, self.nx_model)

                cell_array[x1:x2, y1:y2] = _get_matrix_from_kernel(kernel, nx_node)
            
            self.mapped_bins.append(MappedBin(bin_id, cell_array))


        return 
    
    def plot(self, bin = None, filepath = None, name = None):
        '''
        Plots the packed model as a grid of rectangles
        '''

        pu.plot_packing_efficient(self.packer,filepath=filepath, name=name)
        return
    
    def __repr__(self):
        self.plot()
        return f"NxModelMapping(nbins={self.nbins}, core_size={self.core_size})"

class packed_model(object):
    '''
    Runs bin packing on a cgraph

    > Splits a cgraph's matrices into at most imc_core_size sized matrices.
    > Uses rectangular packing to pack each matrix into imc_core_size arrays
    > IDs of the packed rectangles are the order of the nodes in the cgraph.
    > Only nodes with attribute "matrix" are turned into rectangles.
    '''
    def __init__(self,
                 inshapes, 
                 imc_core_size : tuple[int],
                 infer = False,
                 packer = None,
                 split = True
                 ):
        '''
        Performs bin packing on cgraph and
        initializes aimc cores based on number of bins

        Initializes a numpy array of the core only if a cgraph is input
        '''
        if type(inshapes) != cgraph.Cgraph:
            # Input must be a list
            cgraph_shapes = splitter.split_shapelist_into_chunks(inshapes,*imc_core_size)
            cgraph_shapes = get_ids_for_shapelist(cgraph_shapes)
            packer = rectpack.newPacker(rotation=False,
                                        pack_algo=rectpack.MaxRectsBssf)        
            packer = add_rects_to_packer(packer,cgraph_shapes)

            packer.add_bin(*imc_core_size,count=float("inf"))
            packer.pack()
            self.packer = packer
            return

        # Input is a cgraph
        if split:
            inshapes = cgraph.split_convolutions(inshapes,H=imc_core_size[0],W=imc_core_size[1])

        print('Getting shapes from cgraph')
        cgraph_shapes = inshapes._get_shape_list_id(excludeDepthwise=True)

        if packer is None:
            print('Packer is none. Using default offline MaxRectsBSSF.')
            packer = rectpack.newPacker(mode=rectpack.PackingMode.Offline, rotation=False, pack_algo=rectpack.MaxRectsBssf)

        packer = pack_shapes_into_coresize_bins(packer, cgraph_shapes, imc_core_size)
        check_packing_success(packer, cgraph_shapes)

        self.nbins = len(packer)
        self.cgraph_shapes = cgraph_shapes
        self.bin_matrices = get_mapped_matrices_from_packer(inshapes, packer, imc_core_size)
        self.packer = packer
        self.cgraph = inshapes
        self.core_size = imc_core_size

        return 
    
    @classmethod
    def from_onnx_model(cls,
                 nx_model : onnx.ModelProto,
                 imc_core_size : tuple[int],
                 infer = False,
                 packer = None,
                 **kwargs
                 ):
        '''
        Loads a model from onnx and packs it into imc_core_size sized matrices
        '''
        cgraph_UUT = cgraph.Cgraph.from_onnx_model(nx_model, channel_minor=True)
        return cls(cgraph_UUT,imc_core_size,infer=infer,packer=packer)

    def plot(self, bin = None, filepath = None, name = None):
        '''
        Plots the packed model as a grid of rectangles
        '''

        pu.plot_packing_efficient(self.packer,filepath=filepath, name=name)

        return

    def find_bin_of_id(self, node_id):
        '''
        Finds the bin of a node ID
        '''
        for i,bin in enumerate(self.packer):
            for rect in bin:
                if rect.rid == node_id:
                    return i
        return None

class QRAccModel(object):
    '''
    Analytical model of a qracc accelerator
    '''
    def __init__(self,
                 packed_cgraph : packed_model,
                 num_cores = 1
                 ):
        
        self.packed_cgraph = packed_cgraph
        self.total_bins = len(packed_cgraph.packer)
        self.calculate_utilization()
        self.num_cores = num_cores
        self.predict_weight_rewrites(num_cores)
        self.core_size = packed_cgraph.core_size
        return

    def calculate_utilization(self):

        total_matrix_area = 0
        for bin in self.packed_cgraph.packer:
            for rect in bin:
                total_matrix_area += rect.width * rect.height

        total_core_area = len(self.packed_cgraph.packer) * 256 * 256

        self.utilization = total_matrix_area / total_core_area

    def predict_weight_rewrites(self, num_cores=1): 
        current_bins = {i:None for i in range(num_cores)}
        bin_uses = {i:0 for i in range(num_cores)}
        bin_writes = 1

        # create dictionary of the bins in the packed model
        times_written = {i:0 for i,bin in enumerate(self.packed_cgraph.packer)}

        # create dictionary of write number vs bin ID written
        write_list = {i+1:0 for i in range(len(self.packed_cgraph.packer))}

        for matshape in self.packed_cgraph.cgraph_shapes:
            # get the bin that this matshape is in
            bin_id = self.packed_cgraph.find_bin_of_id(matshape[2])
            if bin_id is None:
                raise ValueError(f"Matshape {matshape} not found in packed model bins")

            # print(f'accessing bin {bin_id}:{current_bins}, {bin_uses}')
            
            if bin_id not in current_bins.values():
                lfu_bin = min(bin_uses, key=bin_uses.get)
                current_bins[lfu_bin] = bin_id
                # reset bin_uses of lfu_bin
                bin_uses[lfu_bin] = 0
                times_written[bin_id] += 1
                bin_writes += 1
                write_list[bin_writes] = bin_id
            else:
                # increment bin_uses of current_bins with value bin_id
                for k, v in current_bins.items():
                    if v == bin_id:
                        bin_uses[k] += 1
                        break

        # print('Total bin writes:', bin_writes)
        # print('Bin uses:', bin_uses)

        self.weight_bin_writes = bin_writes
        self.weight_times_written = times_written
        self.weight_write_list = write_list

        return bin_writes

    def plot_bin_writes_bargraph(self, filepath=None, name=None, ax=None):
        times_written = self.weight_times_written

        if ax is None:
            fig, ax = plt.subplots(figsize=[3, 2], dpi=300)
        
        sns.barplot(x=list(times_written.keys()), y=list(times_written.values()), ax=ax)

        # Label only every N ticks based on figure size
        num_bins = len(times_written)
        if num_bins > 10:
            step = num_bins // 5  # Show ~5 ticks
            ax.set_xticks(ax.get_xticks()[::step])

        ax.set_xlabel("Bin ID")
        ax.set_ylabel("$N_{writes}$")
        ax.set_title(f"{name}")
        
        return ax

    def plot_write_lineplot(self, filepath=None, name=None, ax=None):
        write_list = self.weight_write_list
        if ax is None:
            fig, ax = plt.subplots(figsize=[3, 2], dpi=300)
        sns.lineplot(x=list(write_list.keys()), y=list(write_list.values()), 
                    marker='o', markersize=4, color='blue', linewidth=0.5, ax=ax)

        # sns.jointplot(x=list(write_list.keys()), y=list(write_list.values()),
        #                kind="scatter", ax=ax, color='blue', marker='o', s=20)
        ax.set_xlabel("Write ID") 
        ax.set_ylabel("Bin ID")
        ax.set_title(f"{name}")
        return ax

    def simulate_inference(self):
        return