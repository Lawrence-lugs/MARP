import torch

# core = torch.tensor
   
class core():
    def __init__(self, inbuffersize, outbuffersize, precision):
        ''' The digital and AIMC cores will later inherit from this '''        
        self.in_buffer = torch.zeros([dims[0], inbuffersize], dtype=torch.int8)
        self.out_buffer = torch.zeros([dims[1], outbuffersize], dtype=torch.int8) 
        self.xbar = torch.ones()
        
    def forward
        

class accelerator():
    def __init__(self, corelist):
        ''' Contains a ordered list of AIMC and digital cores, and a connectivity graph from that list  '''
        self.cores = corelist

    def fit(self):
        ''' Attempts to fit a computational graph into the accelerator '''

    def forward(self):
        ''' Performs inference  '''

