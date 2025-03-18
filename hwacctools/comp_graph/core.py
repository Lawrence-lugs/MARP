from . import splitter,cnodes,cgraph
import rectpack
import numpy as np

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

def pack_matrices(cgraph,core_size,packer):
    inshapes = cgraph.split_convolutions(inshapes,H=core_size[0],W=core_size[1])
    cgraph_shapes = inshapes._get_shape_list_id()

    packer = rectpack.newPacker(rotation=False,
                                pack_algo=rectpack.MaxRectsBssf)
    packer = add_rects_to_packer(packer,cgraph_shapes)

    packer.add_bin(*core_size,count=float("inf"))
    packer.pack()
    return cgraph,core_size,packer

class packed_model(object):
    '''
    Runs bin packing on a cgraph

    > Splits a cgraph's matrices into at most core_size sized matrices.
    > Uses rectangular packing to pack each matrix into core_size arrays
    > IDs of the packed rectangles are the order of the nodes in the cgraph.
    > Only nodes with attribute "matrix" are turned into rectangles.
    '''
    def __init__(self,
                 inshapes, 
                 core_size : tuple[int],
                 infer = False
                 ):
        '''
        Performs bin packing on cgraph and
        initializes aimc cores based on number of bins

        Initializes a numpy array of the core only if a cgraph is input
        '''
        if type(inshapes) != cgraph.Cgraph:
            # Input must be a list
            cgraph_shapes = splitter.split_shapelist_into_chunks(inshapes,*core_size)
            cgraph_shapes = get_ids_for_shapelist(cgraph_shapes)
            packer = rectpack.newPacker(rotation=False,
                                        pack_algo=rectpack.MaxRectsBssf)        
            packer = add_rects_to_packer(packer,cgraph_shapes)

            packer.add_bin(*core_size,count=float("inf"))
            packer.pack()
            self.packer = packer
            return

        # Input is a cgraph
        inshapes = cgraph.split_convolutions(inshapes,H=core_size[0],W=core_size[1])
        cgraph_shapes = inshapes._get_shape_list_id(excludeDepthwise=True)

        packer = rectpack.newPacker(rotation=False,
                                    pack_algo=rectpack.MaxRectsBssf)
        packer = add_rects_to_packer(packer,cgraph_shapes)

        packer.add_bin(*core_size,count=float("inf"))
        packer.pack()

        if len(packer.rect_list()) != len(cgraph_shapes):
            print(f'Packing incomplete: packed {len(packer.rect_list())} vs {len(cgraph_shapes)} matrices in cgraph')
        else:
            print(f'Packing successful: packed {len(packer.rect_list())} vs {len(cgraph_shapes)} matrices in cgraph')


        self.ncores = len(packer)
        self.cores = []

        for bin in packer:

            # Prepare matrix for a new core
            cell_array = np.zeros(core_size)

            # Populate it with the mapped matrices 
            for mapped_rect in bin:
                mapped_node = inshapes.nodes[mapped_rect.rid]

                x1 = mapped_rect.x
                x2 = mapped_rect.corner_top_r.x
                y1 = mapped_rect.y
                y2 = mapped_rect.corner_top_r.y

                cell_array[y1:y2,x1:x2] = mapped_node.matrix

            self.cores.append( matmul_core(core_size,cell_array) )

        self.packer = packer
        self.cgraph = inshapes

        return 

class matmul_core(object):
    '''
    matmul core that does everything a matmul core can do and no more than that.

    TODO
    '''

    def __init__(self,
                 core_size : tuple[int],
                 matrix : np.ndarray
                 ):
        '''
        Parameters
        ----------
        core_size : tuple(int,int),
            dimensions of bitcell array
        matrix : np.ndarray,
            weights loaded into bitcell array
        '''
        self.cell_array = matrix
        self.input_buffer = np.empty( core_size[0] )    
        self.output_buffer = np.empty( core_size[1] )
        return
