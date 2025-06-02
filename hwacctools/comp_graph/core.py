from . import splitter,cnodes,cgraph, packer_utils  as pu
import numpy as np
import onnx
import matplotlib.pyplot as plt
import rectpack
import seaborn as sns

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