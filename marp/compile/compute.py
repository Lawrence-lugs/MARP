import numpy as np

def get_recfield_for_pixel(r, c, matrix, kernel_shape):
    'Obtains the receptive field for a convolution output pixel with arbitrary kernel shape'
    kh, kw = kernel_shape
    if kh == 1 and kw == 1:
        return matrix[r, c]
    else:
        # General case for any kernel shape
        return matrix[r:r+kh, c:c+kw, :]

def toeplitzize_input(in_tensor, ksize=3, strides=1, channel_minor=False, zero_point=0, pads=(1,1,1,1), kernel_shape=None):
    '''
    Flattens input tensor into a Toeplitz matrix for passing into a
    flattened kernel. Zero pads by default.

    input: B,C,H,W tensor

    Assumes B=1 for now
    '''
    if kernel_shape is None:
        kernel_shape = (ksize, ksize)
    kk, kc, kh, kw = kernel_shape

    # Convert to B,H,W,C tensor
    tensor = in_tensor.transpose(1, 2, 0)
    ifmap_shape = tensor.shape

    if type(strides) is int:
        stridesx = strides
        stridesy = strides
    else:
        stridesx = strides[0]
        stridesy = strides[1]

    H = (tensor.shape[0] - kh + 2 * pads[0]) // stridesx + 1
    W = (tensor.shape[1] - kw + 2 * pads[1]) // stridesy + 1
    C = tensor.shape[2]

    pad_width = ((pads[0], pads[1]), (pads[2], pads[3]), (0, 0))
    tensor2 = np.pad(tensor, pad_width, mode='constant', constant_values=zero_point)
    out = np.empty((H * W, C * kh * kw), dtype=tensor.dtype)

    for r in range(H):
        for c in range(W):
            recfield = get_recfield_for_pixel(stridesy * r, stridesx * c, tensor2, (kh, kw))
            if channel_minor and (kh > 1 or kw > 1):
                # Move channel to last axis, then flatten
                out[r * W + c] = np.transpose(recfield, (1, 2, 0)).flatten()
            else:
                out[r * W + c] = recfield.flatten()

    return out
