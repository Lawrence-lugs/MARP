#%%

from rectpack import newPacker

rectangles = [(100, 30), (40, 60), (30, 30),(70, 70), (100, 50), (30, 30)]
bins = [(100, 100), (100, 100), (100, 100), (100,100), (100,100)]

packer = newPacker()

#packing queue
for r in rectangles:
    packer.add_rect(*r)

# we can add rectangle IDs as follows
# packer.add_rect(width,height[,rid]) 

for b in bins:
    packer.add_bin(*b)

packer.pack()

#%%

nbins = len(packer)

abin = packer[0]

for bin in packer:
    print(bin.rect_list())

a = 1