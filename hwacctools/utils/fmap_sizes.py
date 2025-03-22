from hwacctools.comp_graph import splitter, cnodes, cgraph
import numpy as np

def try_len_flatten(i):
    try:
        return len(i.flatten())
    except:
        return 0

def get_feature_sizes(cgraph):
    feature_sizes = { k:try_len_flatten(v) for (k,v) in cgraph.edges.items()}
    # clean up
    keys_to_remove = []
    for k,v in feature_sizes.items():
        if v == 0:
            keys_to_remove.append(k)
        # remove the pre-scalers (those don't go into activation memory)
        if 'scaler_input' in k:
            keys_to_remove.append(k)
    for k in keys_to_remove:
        feature_sizes.pop(k)            
    return feature_sizes

def get_fmap_pair_sizes(cgraph):
    feature_sizes = get_feature_sizes(cgraph)
    
    prev_edge = ''
    prev_fmap_size = 0
    concurrent_fmap_sizes = {}
    for edgename, fmap_size in feature_sizes.items():
        concurrent_fmap_sizes[f'{prev_edge} to {edgename}'] = fmap_size + prev_fmap_size

        prev_fmap_size = fmap_size
        prev_edge = edgename

    return concurrent_fmap_sizes

def get_largest_fmap(cgraph):
    feature_sizes = get_feature_sizes(cgraph)
    largest_fmap = max(feature_sizes, key=feature_sizes.get), feature_sizes[max(feature_sizes,key=feature_sizes.get)]
    return largest_fmap

def get_largest_fmap_pair(cgraph):
    ' for pingpong purposes '

    concurrent_fmap_sizes = get_fmap_pair_sizes(cgraph)

    largest_concurrent_fmap = max(concurrent_fmap_sizes, key=concurrent_fmap_sizes.get), concurrent_fmap_sizes[max(concurrent_fmap_sizes,key=concurrent_fmap_sizes.get)]
    return largest_concurrent_fmap

if __name__ == '__main__':
    mem_size_needed = get_largest_fmap_pair(u_packed.cgraph)[1] * 8
    print(get_largest_fmap(u_packed.cgraph), get_largest_fmap_pair(u_packed.cgraph), mem_size_needed)
