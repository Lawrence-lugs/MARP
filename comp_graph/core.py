import splitter,cnodes,cgraph
import rectpack
import numpy as np

class aimc_acc(object):
    '''
    Set of AIMC cores for simulation
    '''
    def __init__(self,
                 cgraph : cgraph.cgraph, 
                 core_size : tuple[int]
                 ):
        '''
        Performs bin packing on cgraph and
        initializes aimc cores based on number of bins
        '''
        cgraph_shapes = cgraph._get_shape_list_id()
        
        packer = rectpack.newPacker(rotation=False,
                                    pack_algo=rectpack.MaxRectsBssf)

        for shape in cgraph_shapes:
            # Shape in numpy is y,x but we want x,y
            rectw = shape[1]
            recth = shape[0]
            rid = shape[2]
            packer.add_rect(rectw,recth,rid)

        packer.add_bin(*core_size,count=float("inf"))
        packer.pack()
        self.ncores = len(packer)
        self.cores = []

        for bin in packer:

            # Prepare matrix for a new core
            cell_array = np.zeros(core_size)

            # Populate it with the mapped matrices 
            for mapped_rect in bin:
                mapped_node = cgraph.nodes[mapped_rect.rid]

                x1 = mapped_rect.x
                x2 = mapped_rect.corner_top_r.x
                y1 = mapped_rect.y
                y2 = mapped_rect.corner_top_r.y

                cell_array[y1:y2,x1:x2] = mapped_node.matrix

            self.cores.append( aimc_core(core_size,cell_array) )

        self.packer = packer

        return 

class aimc_core(object):
    '''
    AIMC core with a limited buffer
    No need for pipeline registers for now
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
    
    def load_dac_buffer(self,
                        input_vector : np.ndarray):
        self.output_pipe_regs = input_vector @ self.cell_array
        return
