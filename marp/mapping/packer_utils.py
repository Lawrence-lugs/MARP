#%%

import rectpack
import matplotlib.pyplot as plt
from matplotlib import patches
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

def get_packer_by_type(packername):
    if packername == 'Naive':
        return NaiveRectpackPacker(256, 256)
    elif packername == 'Dense':
        return rectpack.newPacker(
            mode=rectpack.PackingMode.Offline,
            bin_algo=rectpack.PackingBin.BBF, 
            rotation=False, 
            pack_algo=rectpack.MaxRectsBssf
        )
    elif packername == 'Balanced':
        return rectpack.newPacker(
            mode=rectpack.PackingMode.Online,
            bin_algo=rectpack.PackingBin.BBF, 
            rotation=False, 
            pack_algo=rectpack.MaxRectsBssf
        )
    elif packername == 'WriteOptimized':
        return rectpack.newPacker(
            mode=rectpack.PackingMode.Online,
            bin_algo=rectpack.PackingBin.BNF, 
            rotation=False, 
            pack_algo=rectpack.MaxRectsBssf
        )
    else:
        raise ValueError(f"Unknown packer name: {packername}")

def get_packer(naive : bool, offline : bool, bin_algo : str, pack_algo : str, sort_algo : str, core_size : tuple = None):

    bin_algo_map = {
        'BBF': rectpack.PackingBin.BBF,
        'BFF': rectpack.PackingBin.BFF,
        'BNF': rectpack.PackingBin.BNF
    }
    pack_algo_map = {
        'MaxRectsBssf': rectpack.MaxRectsBssf,
        'MaxRectsBaf': rectpack.MaxRectsBaf,
        'SkylineBlWm': rectpack.SkylineBlWm
    }
    sort_algo_map = {
        'SORT_AREA': rectpack.SORT_AREA,
        'SORT_PERI': rectpack.SORT_PERI,
        'SORT_DIFF': rectpack.SORT_DIFF,
        'SORT_SSIDE': rectpack.SORT_SSIDE,
        'SORT_LSIDE': rectpack.SORT_LSIDE,
        'SORT_RATIO': rectpack.SORT_RATIO,
        'SORT_NONE': rectpack.SORT_NONE
    }
    mode = rectpack.PackingMode.Online if not offline else rectpack.PackingMode.Offline

    bin_algo_list = list(bin_algo_map.keys())
    pack_algo_list = list(pack_algo_map.keys())
    sort_algo_list = list(sort_algo_map.keys())

    if bin_algo not in bin_algo_list:
        raise ValueError(f"Invalid bin_algo: {bin_algo}. Must be one of {bin_algo_list}")
    if pack_algo not in pack_algo_list:
        raise ValueError(f"Invalid pack_algo: {pack_algo}. Must be one of {pack_algo_list}")
    if sort_algo not in sort_algo_list: 
        raise ValueError(f"Invalid sort_algo: {sort_algo}. Must be one of {sort_algo_list}")
    
    if naive and core_size is not None:
        return NaiveRectpackPacker(core_size[0], core_size[1], rotation=False)
    elif naive:
        return NaiveRectpackPacker(256, 256, rotation=False)

    return rectpack.newPacker(
        bin_algo=bin_algo_map[bin_algo],
        pack_algo=pack_algo_map[pack_algo],
        sort_algo=sort_algo_map[sort_algo],
        rotation=False,
        mode=mode
    )

def plot_packing_efficient(packer, tile_count_h = 6, filepath=None, name = None, **kwargs):
    '''
    Plots the packed bins in a grid layout with pastel colors.
    '''
    pastel_colors = [
        "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
        "#FFABAB", "#FFC3A0", "#D5AAFF", "#C2F0C2", "#B2B2B2"
    ]

    if len(packer) > tile_count_h:
        n_subplot_cols = tile_count_h
        n_subplot_rows = (len(packer) + n_subplot_cols - 1) // n_subplot_cols
    else:
        n_subplot_cols = np.sqrt(len(packer)).astype(int)
        n_subplot_rows = np.ceil(len(packer) / n_subplot_cols).astype(int)

    fig, axs = plt.subplots(n_subplot_cols, n_subplot_rows, figsize=(9, 9))
    if n_subplot_rows == 1 and n_subplot_cols == 1:
        axs = np.array([[axs]])

    for index,abin in enumerate(packer):
        # print each rectangle inside packed bin in a plot
        
        plt.subplot(n_subplot_cols,n_subplot_rows, index + 1)

        for i_rect,rect in enumerate(abin):
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            # print(i_rect % len(pastel_colors))
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, facecolor=pastel_colors[i_rect % len(pastel_colors)],edgecolor='#000000', linewidth=2))
            plt.xlim(0, abin.width)
            plt.ylim(0, abin.height)
            
        # add title with bin index
        plt.title(f"{index + 1}", fontsize=10)

    # plt.tight_layout()
    # add overall title
    if name is not None:
        plt.suptitle(f"{name}", fontsize=16, y=0.95)

    # turn off axis ticks for all subplots
    for ax in axs.flat:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axis('off')

    # Add legend of pastel colors by index order
    if kwargs.get('legend', False):
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in pastel_colors]
        labels = [f"{i}" for i in range(len(pastel_colors))]
        plt.figlegend(handles, labels, loc='upper right', fontsize='medium', title='Packing order')
    

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    return 


def plot_bins(packer,
                 dir : str
                 ):
    '''
    From rectpack issue 16

    Use to create single images of each bin.
    Does not create a unified image.
    '''
    
    if not os.path.exists(f'{dir}'):
        os.mkdir(f'{dir}')

    for index, abin in tqdm(enumerate(packer),desc='Plotting Bins'):
        bw, bh  = abin.width, abin.height
        # print('bin', bw, bh, "nr of rectangles in bin", len(abin))
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        for rect in abin:
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            plt.axis([0,bw,0,bh])
            # print('rectangle', w,h)
            ax.add_patch(
                patches.Rectangle(
                    (x, y),  # (x,y)
                    w,          # width
                    h,          # height
                    facecolor="#00ffff",
                    edgecolor="black",
                    linewidth=3
                )
            )
    
            center_x = x + w/2
            center_y = y + h/2
            ax.annotate(str(rect.rid),(center_x,center_y))
        fig.savefig(f'{dir}/rect_{index}.png', dpi=144, bbox_inches='tight')

def plot_packing(packer,
                 filename : str,
                 tile_count_h : int = 8,
                 color_dict = None,
                 rid_to_nid = None
                 ):
    '''
    Plots the bins together and saves a single image
    '''
    result = None

    for index, abin in tqdm(enumerate(packer),desc='Plotting Bins'):
        bw, bh  = abin.width, abin.height
        # print('bin', bw, bh, "nr of rectangles in bin", len(abin))
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        for rect in abin:
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            plt.axis([0,bw,0,bh])
            # print('rectangle', w,h)

            facecolor = "#f7dc6f"
            edgecolor = "#ca6f1e"

            if color_dict is not None:
                if rect.rid in color_dict.keys():
                    facecolor = color_dict[rect.rid]

            ax.add_patch(
                patches.Rectangle(
                    (x, y),  # (x,y)
                    w,          # width
                    h,          # height
                    facecolor=facecolor,
                    edgecolor="black",
                    linewidth=3
                )
            )                
 
            center_x = x + w/2
            center_y = y + h/2

            if rid_to_nid is not None:
                ax.annotate(str(rid_to_nid[rect.rid]),(center_x,center_y))
            else:
                ax.annotate(str(rect.rid),(center_x,center_y))

                
        # fig.savefig(f'figures/{dir}/rect_{index}.png', dpi=144, bbox_inches='tight')
        fig.canvas.draw()
        figwh = fig.canvas.get_width_height()

        if result is None:
            tile_count_v = len(packer) // tile_count_h + (len(packer) % tile_count_h != 0)
            res_img_h = figwh[1] * tile_count_v #ceil_func
            res_img_w = figwh[0] * tile_count_h
            result = Image.new("RGB",(res_img_w,res_img_h))

        bin_img_bytes = fig.canvas.tostring_rgb()
        bin_img = Image.frombytes('RGB',figwh,bin_img_bytes) 

        tile_x = index % tile_count_h
        tile_y = index // tile_count_h

        result.paste(bin_img, (tile_x * figwh[0], tile_y * figwh[1]))

        plt.close(fig)

    result.save(f'{filename}.png')

def combine_bin_pictures(name : str, 
                         path = './bin_figures',
                         horizontal_tile_count=8
                         ):
    '''
    Combines the rect_<rid> pictures into a single large figure
    '''
    bin_files = os.listdir(path)

    num_bins = len(bin_files)
    ref_image = Image.open(path + bin_files[0])
    tile_dim = ref_image.size

    num_tile_rows = num_bins // horizontal_tile_count + (num_bins % horizontal_tile_count !=0) #ceil_func
    out_image_w = horizontal_tile_count * tile_dim[0]
    out_image_h = tile_dim[1] * num_tile_rows

    result = Image.new("RGBA",(out_image_w,out_image_h))

    for i, bin_filename in enumerate(bin_files):
        img = Image.open(path + bin_filename)        

        tile_x = i % horizontal_tile_count
        tile_y = i // horizontal_tile_count

        # print(tile_x,tile_y)

        result.paste(img, (tile_x * tile_dim[0], tile_y * tile_dim[1] ))

    result.save(name + '.png')

class NaiveRectpackPacker(object):
    '''
    Naive rectpack packer that packs one rectangle in each bin.
    For example, if you have 10 rectangles, it will create 10 bins
    with one rectangle each. This class mimics the behavior of a
    rectpack.Packer object.
    '''
    def __init__(self, bin_width, bin_height, rotation=True):
        self.bin_width = bin_width
        self.bin_height = bin_height
        self._rectangles = []
        self._bins = []
        self.rotation = rotation

    def add_rect(self, width, height, rid=None):
        self._rectangles.append({'width': width, 'height': height, 'rid': rid})

    def add_bin(self, width, height, count=1):
        # This packer's logic is one rect per bin, so bin size is set at init.
        # This method is for API compatibility.
        pass

    def pack(self):
        self._bins = []
        for i, r in enumerate(self._rectangles):
            # Create a new packer and bin for each rectangle
            p = rectpack.newPacker(rotation=self.rotation)
            p.add_bin(self.bin_width, self.bin_height)
            p.add_rect(r['width'], r['height'], rid=r['rid'])
            p.pack()
            
            # There will be exactly one bin, and one rectangle in it.
            if len(p) > 0:
                self._bins.append(p[0])

    def rect_list(self):
        return self._rectangles

    def __iter__(self):
        return iter(self._bins)

    def __getitem__(self, key):
        return self._bins[key]

    def __len__(self):
        return len(self._bins)

#%%

if __name__ == '__main__':
    combine_bin_pictures('all_bins','bin_figures/',8)
# %%
