
#%%

import numpy as np
import matplotlib.pyplot as plt
import objects
import model_flattener

def remove_depthwise(matrices):
    return [i for i in matrices if len(i.shape) == 2]

def split_to_fit_in_WH(matrices,W,H):
    out = []
    for i,mat in enumerate(matrices):
        if mat.shape[0] > W:
            reps = mat.shape[0]//W + (mat.shape[0]%W!=0) #ceil func
            for i in range(reps):
                out.append(mat[W*i:W*i+W])
        else:
            out.append(mat)

    out2 = []
    for i,mat in enumerate(out):
        if mat.shape[1] > H:
            reps = mat.shape[1]//H + (mat.shape[1]%H!=0) #ceil func
            for i in range(reps):
                out2.append(mat[:,H*i:H*i+H])
        else:
            out2.append(mat)
    return out2

def non_increasing_sort(matrix_set):
    '''
    That is, decreasing but may be equal
    '''
    sizes = np.array(model_flattener.get_shapes(matrix_set))
    order = np.lexsort((sizes[:,1],sizes[:,0]))[::-1]
    matrices = np.array(matrix_set,dtype=object)[order]
    sizes = sizes[order]

    return order, sizes, matrices

def plot_placements(matrices, placements, width, height):
    canvas = np.zeros((height,width))
    for i,placement in enumerate(placements):
        left = placement[0]
        right = left + matrices[i].shape[1]
        bottom = placement[1]
        top = bottom + matrices[i].shape[0]
        canvas[bottom:top,left:right] = i%4 + 1
    plt.figure(figsize=(4,40))
    plt.imshow(canvas[::-1])
    plt.axis('off')
    plt.axis('tight')
    plt.axis('image')
    plt.show()
    return canvas

def ffd_strip(in_shapes, bin_W):
    '''
    First Fit Decreasing Strip-Packing Algorithm
    '''
    order, shapes, matrices = non_increasing_sort(in_shapes)
    
    # limit = 90
    # shapes = shapes[-limit:]
    # matrices = matrices [-limit:]

    # corners in x,y
    avail_corners = np.array([[0,0]],int)

    placements = np.empty((0,2),int)

    # place first shape

    def fits_in_level(shape,corner):
        return shape[1] <= bin_W - corner[0]

    # shapes are in y, x (numpy matrix convention)
    for k,shape in enumerate(shapes):
        for i,corner in enumerate(avail_corners):

            if fits_in_level(shape,corner):

                placements = np.vstack([placements,[corner]])
                
                # if shape is the first one in the level, add new level
                if corner[0] == 0:
                    avail_corners = np.vstack([avail_corners,[0,corner[1]+shape[0]]])

                avail_corners[i][0] += shape[1]
                
                break

        # print(f'Placements,: \n{placements[-5:]}')
        # print(f'Corners: \n{avail_corners[-5:]}')
        # print(f"Shape mapped: \n{k,shape} at {placements[-1]}")

    highest_placement_index = placements.argmax(axis=0)[1]
    strip_height = placements[highest_placement_index][1] + matrices[highest_placement_index].shape[0]
    
    plot_placements(matrices,placements,bin_W,strip_height)

    placements_ordered = placements[order]

    return placements_ordered, strip_height

def hff(shapes, bin_W, bin_H):
    '''
    Hybrid First Fit algorithm for 2D Bin Packing
    '''

if __name__ == '__main__':

    import torch
    from torchvision import models
    
    import model_flattener

    mbv2 = models.mobilenet_v2(pretrained=True)

    with torch.no_grad():
        strides,mbv2_flat = model_flattener.matricize_model(mbv2)

    bin_width = 256
    bin_height = 256
    #shapes = model_flattener.get_shapes_for_2BP(mbv2_flat,bin_width,bin_height)

    mbv2_flat_nodw = remove_depthwise(mbv2_flat)
    mbv2_for_binning = split_to_fit_in_WH(mbv2_flat_nodw,bin_height,bin_width)
    mbv2_4bin_sizes = np.array(model_flattener.get_shapes(mbv2_for_binning))

    n_params = model_flattener.count_parameters(mbv2_for_binning)
    print(f'Number of parameters: {n_params}')

    placements, strip_height = ffd_strip(mbv2_for_binning,bin_width)

    cores_used = strip_height//bin_height + (strip_height%bin_height!=0) #ceil func

    print(f'Strip Height: {strip_height}')
    print(f'{bin_width}x{bin_height} Cores used: {cores_used}')

# %%