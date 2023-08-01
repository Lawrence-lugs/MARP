#%%

import numpy as np
import matplotlib.pyplot as plt
import objects

core = objects.core
accelerator = objects.accelerator

def sort_matrices_by_height(matrix_list):
    sizes = []
    for matrix in matrix_list:
        sizes.append(matrix.shape)
    sizes = np.array(sizes)
    return matrix_list[sizes[:,0].argsort()[::-1]]

matrix_sizes = np.array([
    (32,32),
    (16,24),
    (64,32),
    (23,12),
])

matrix_sizes = np.array([
    (32,32),
    (16,64),
    (16,128),
    (16,16),
    (64,32),
    (64,64),
    (92,64),
    (92,128)
])

matrix_list = []
for size in matrix_sizes:
    matrix_list.append(np.random.rand(*size))
matrix_list = np.array(matrix_list)

matrix_list =(sort_matrices_by_height(matrix_list))

sheetdims = (256,256)
sheet = core(sheetdims)

for matrix in matrix_list:
    sheet.map(matrix)
    sheet.show()
    plt.show()
    plt.figure()
    import pdb
    pdb.set_trace()

sheet.show()

#%%

    
b = accelerator()
        
matrix_list = [
    (32,32),
    (16,64),
    (16,128),
    (16,16),
    (64,32),
    (64,64),
    (92,64),
    (92,128)
]
for matrix in matrix_list:
    if accelerator.map(matrix) == None:
        print("Matrix no longer fits")
        break
    accelerator.show()


# %%
