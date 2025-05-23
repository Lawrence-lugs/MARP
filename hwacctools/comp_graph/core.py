from . import splitter,cnodes,cgraph
import rectpack
import numpy as np
import onnx

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
                 packer = None
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
        inshapes = cgraph.split_convolutions(inshapes,H=imc_core_size[0],W=imc_core_size[1])
        cgraph_shapes = inshapes._get_shape_list_id(excludeDepthwise=True)

        if packer is None:
            print('Packer is none. Using default offline MaxRectsBSSF.')
            packer = rectpack.newPacker(mode=rectpack.PackingMode.Offline, rotation=False, pack_algo=rectpack.MaxRectsBssf)

        packer.add_bin(*imc_core_size,count=float("inf"))
        packer = add_rects_to_packer(packer,cgraph_shapes)
        if hasattr(packer,'pack'):
            packer.pack()

        if len(packer.rect_list()) != len(cgraph_shapes):
            print(f'Packing incomplete: packed {len(packer.rect_list())} vs {len(cgraph_shapes)} matrices in cgraph')
        else:
            print(f'Packing successful: packed {len(packer.rect_list())} vs {len(cgraph_shapes)} matrices in cgraph in {len(packer)} bins')

        self.nbins = len(packer)
        self.cgraph_shapes = cgraph_shapes

        self.bin_matrices = []
        for bin in packer:

            # Prepare matrix for a new core
            cell_array = np.zeros(imc_core_size)

            # Populate it with the mapped matrices 
            for mapped_rect in bin:
                mapped_node = inshapes.nodes[mapped_rect.rid]

                x1 = mapped_rect.x
                x2 = mapped_rect.corner_top_r.x
                y1 = mapped_rect.y
                y2 = mapped_rect.corner_top_r.y

                cell_array[y1:y2,x1:x2] = mapped_node.matrix

            self.bin_matrices.append(cell_array)

        self.packer = packer
        self.cgraph = inshapes

        return 
    
    @classmethod
    def from_onnx_model(cls,
                 nx_model : onnx.ModelProto,
                 imc_core_size : tuple[int],
                 infer = False
                 ):
        '''
        Loads a model from onnx and packs it into imc_core_size sized matrices
        '''
        cgraph_UUT = cgraph.Cgraph.from_onnx_model(nx_model, channel_minor=True)
        return cls(cgraph_UUT,imc_core_size,infer=infer)

class accelerator(object):
    '''
    Forward-pass simulation on specific simulation parameters
    '''
    def __init__(self,
                 cgraph : cgraph.Cgraph,
                 imc_core_size : tuple[int],
                 num_imc_cores : int,
                 rs_core_size : tuple[int],
                 num_rs_cores : int
                 ):
        
        self.original_cgraph = cgraph
        self.imc_core_size = imc_core_size
        self.num_imc_cores = num_imc_cores
        self.rs_core_size = rs_core_size
        self.num_rs_cores = num_rs_cores
        self.packed_model = packed_model(cgraph,imc_core_size)
        return
    
    def simulate_inference(self):

        imc_core_loads = np.zeros(self.num_imc_cores)
        rs_core_loads = np.zeros(self.num_rs_cores)

        # Let's put this on hold for now, because I don't know how to map depthwise to the RS core yet!        

        return

class row_stationary_core(object):
    '''
    Row stationary accelerator model

    TODO: Model properly based on dimension
    '''
    def __init__(self,
                 kernel,
                 pe_array_size : tuple[int]
                 ):
        self.kernel = kernel
        return

    def __call__(self, input_matrix : np.ndarray):
        '''
        Parameters
        ----------
        input_matrix : np.ndarray,
            input matrix to be convolved
        '''
        return

class imc_accelerator_core(object):
    '''
    Contains operations performable by a single accelerator core
    '''
    def __init__(self,
                 imc_core_size : tuple[int],
                 matrix : np.ndarray
                 ):
        '''
        Parameters
        ----------
        imc_core_size : tuple(int,int),
            dimensions of bitcell array
        matrix : np.ndarray,
            weights loaded into bitcell array
        '''
        self.matrix = matrix
        self.imc_core_size = imc_core_size
        return
    
    def load_matrix(self,matrix):
        '''
        Loads a matrix into the core
        '''
        self.matrix = matrix
        self.loads += 1
        return
    
