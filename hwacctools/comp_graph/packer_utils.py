#%%

import rectpack
import matplotlib.pyplot as plt
from matplotlib import patches
import os
from tqdm import tqdm

def plot_packing_img(packer,
                 dir : str
                 ):
    '''
    From rectpack issue 16
    '''
    
    if not os.path.exists(f'figures/{dir}'):
        os.mkdir(f'figures/{dir}')

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
        fig.savefig(f'figures/{dir}/rect_{index}.png', dpi=144, bbox_inches='tight')

def plot_packing_tiled(packer,
                 filename : str,
                 tile_count_h : int = 8,
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

from PIL import Image

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

#%%

if __name__ == '__main__':
    combine_bin_pictures('all_bins','bin_figures/',8)
# %%
