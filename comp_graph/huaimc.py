import cgraph

def matrix_dict_from_cgraph(input_cgraph:cgraph.Cgraph):
    mx_dict = dict()
    for i,node in enumerate(input_cgraph.nodes):
        if hasattr(node,'matrix'):
            mx_dict[i] = node.matrix
    return mx_dict