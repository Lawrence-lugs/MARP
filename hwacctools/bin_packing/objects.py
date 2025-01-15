#%%

import numpy as np
import matplotlib.pyplot as plt

class accelerator:
    def __init__(self,num_cores = 16,core_sizes = [(256,256)]):
        self.num_cores=num_cores
        self.cores = []
        for xysize in core_sizes:
            print(xysize)
            self.cores.append(core(xysize))

    def map(self,matrix):
        #map in first possible slot
        for core in self.cores: 
            if core.map(matrix) == None:
                continue
            else:
                return core
        return None

class mapped_matrix:
    def __init__(self,matrix,top_left):
        self.top_left = top_left
        self.matrix = matrix
        self.bottom_right = tuple(np.add(top_left,matrix.shape))
        self.top_right = (self.top_left[0],self.bottom_right[1])
        self.bottom_left = (self.bottom_right[0],self.top_left[1])

class core:
    '''
    Core with extra attributes "bottom rights" and "available spots" for implementing
    the optimal 2D rectangular packing solution.
    '''
    def __init__(self,xydims):
        self.width = xydims[0]
        self.height = xydims[1]
        self.mapped_matrices = []
        self.bottom_rights = [(0,0),]
        self.available_spots = [(0,0),]

    def map(self,matrix):
        # first fit lower left
        for i,spot in enumerate(self.available_spots):
            if(self.fits(matrix,spot)):
                self.add_matrix(matrix,spot)
                return True
        return False

    def fits(self,matrix,spot):
        if(matrix.shape[0] > self.width - spot[0]):
            return False
        if(matrix.shape[1] > self.height - spot[1]):
            return False
        return True

    def add_bottom_right(self,corner):

        def in_upper_left_of(i,ref_br):
            if (i == ref_br).all():
                return False
            return (i[0] <= ref_br[0] and i[1] <= ref_br[1])

        self.bottom_rights = np.append(self.bottom_rights,[self.mapped_matrices[-1].bottom_right],axis=0)
        for ref_br in self.bottom_rights:
            self.bottom_rights = [i for i in self.bottom_rights if not in_upper_left_of(i,ref_br)]
        br = np.array(self.bottom_rights)
        self.bottom_rights = br[br[:,0].argsort()]
        return

    def add_matrix(self,matrix,spot):
        print(self.available_spots)
        self.mapped_matrices.append(mapped_matrix(matrix,spot))
        self.add_bottom_right(self.mapped_matrices[-1].bottom_right)
        self.update_available_spots()

    def update_available_spots(self):
        
        brs = np.array(self.bottom_rights)
        if not brs.any():
            return

        self.available_spots = []
        self.available_spots.append((0,brs[0][1]))
        
        if brs.shape[0] == 1: 
            # if there's just one matrix, the next available spot is on the wall
            self.available_spots.append((brs[-1][0],0)) 
            return

        for i,node in enumerate(brs):
            if i == brs.shape[0]-1:
                self.available_spots.append((brs[-1][0],0))
                break
            self.available_spots.append((brs[i][0],brs[i+1][1]))

        avs = np.array(self.available_spots)
        self.available_spots = avs[avs[:,0].argsort()]
        return

    def show(self):
        canvas = np.zeros((self.width,self.height))
        for color,mmatrix in enumerate(self.mapped_matrices):
            canvas[mmatrix.top_left[0]:mmatrix.bottom_right[0],\
                   mmatrix.top_left[1]:mmatrix.bottom_right[1]] = color+1
        plt.imshow(canvas)
        self.canvas = canvas
