
# Transferred in from https://github.com/Lawrence-lugs/eyeriss
# Ended up useful for a bunch of other work

import numpy as np
from scipy.signal import convolve2d

class quantized_tensor:
    '''
    Parameters:
    shape : tuple -- Shape of the tensor
    precision : int -- Number of bits to quantize to
    mode : str -- Quantization mode, 'symmetric', 'maxmin' or '3sigma'
    real_values : np.ndarray -- Real values of the tensor
    quantized_values : np.ndarray -- Quantized values of the tensor
    scale : float -- Scale of the quantization
    zero_point : float -- Zero point of the quantization
    
    Main attributes:
    real_values : np.ndarray -- Real values of the tensor
    quantized_values : np.ndarray -- Quantized values of the tensor
    scale : float -- Scale of the quantization
    zero_point : float -- Zero point of the quantization
    shape : tuple -- Shape of the tensor
    fake_quantized_values : np.ndarray -- Quantized values translated back to real range
    
    Quantized Tensor
    Reals are normalized to [-1,1]
    If mode is 'symmetric', zero point is 0 and scale is 2 / (2**precision - 1)
    
    '''

    def __init__(self,shape=None,precision=None,mode='symmetric',real_values=None,quantized_values=None,scale=None,zero_point=None):
        '''
        '''
        if real_values is not None:
            if precision is None:
                raise ValueError('Precision must be provided')
            self.real_values = real_values
            self.quantize(precision,mode,zero_point=zero_point)
            self.dequantize()
        elif quantized_values is not None:
            if scale is None or zero_point is None:
                raise ValueError('Scale and zero point must be provided')
            self.real_values = None
            self.quantized_values = quantized_values
            self.scale = scale
            self.zero_point = zero_point
            self.dequantize()
        else:
            if shape is None or precision is None:
                raise ValueError('Shape and precision must be provided')
            self.real_values = np.random.uniform(-1,1,shape)
            self.quantize(precision,mode)
            self.dequantize()
        self.shape = self.real_values.shape
        return

    def quantize(self, precision, mode='symmetric', zero_point = None):
        if mode == 'maxmin' : 
            clip_high = self.real_values.max()
            clip_low = self.real_values.min()
            self.scale = 2*max(clip_high,clip_low) / (2**precision - 1)
            if zero_point is None:
                self.zero_point = self.real_values.mean()
            else:
                self.zero_point = zero_point
        elif mode == '3sigma' :
            mean = self.real_values.mean()
            std = self.real_values.std()
            self.scale = std*3 / (2**precision - 1)
            self.zero_point = mean
        elif mode == 'symmetric':
            self.scale = 2 / (2**precision - 1)
            self.zero_point = 0

        # r = S(q - Z)
        self.quantized_values = np.round( (self.real_values / self.scale) + self.zero_point ).astype(int)

    def dequantize(self):
        " Creates fake quantization values from the quantized values, like EdMIPS "
        self.fake_quantized_values = (self.quantized_values - self.zero_point) * self.scale
        if self.real_values is None:
            self.real_values = self.fake_quantized_values
        return

def convolve_fake_quantized(a,w) -> np.ndarray:
    " o = aw "
    return convolve2d(a.fake_quantized_values,w.fake_quantized_values[::-1].T[::-1].T,mode='valid')

def convolve_reals(a,w) -> np.ndarray:
    " o = aw "
    return convolve2d(a.real_values,w.real_values[::-1].T[::-1].T,mode='valid')

def scaling_quantized_matmul(w,a,outBits,internalPrecision,out_scale = None) -> quantized_tensor:
    " o = w @ a.T but quantized "

    # Matrix multiplication
    qaqw = w.quantized_values @ a.quantized_values.T

    # Scaling
    if out_scale is None:
        out_scale = 2 / (2**outBits - 1)
    newscale = a.scale * w.scale / out_scale
    m0, shift = convert_scale_to_shift_and_m0(
                    newscale,
                    precision=internalPrecision
                )

    fp_m = m0 * 2**(shift)
    # m0bin = convert_to_fixed_point(m0, internalPrecision)
    # m0int = int(m0bin,base=2)

    scaled_clipped_shifted = (qaqw * fp_m).astype(int)
    o_q = saturating_clip(scaled_clipped_shifted, outBits)

    # Quantized tensor of the output
    o_qtensor = quantized_tensor(
                    qaqw.shape,
                    outBits,
                    quantized_values=o_q,
                    scale=out_scale,
                    zero_point=0 # Assume 0 for now, might be bad later.
                )

    return o_qtensor

def scaling_quantized_convolution(a,w,outBits,internalPrecision, out_scale = None) -> quantized_tensor:
    " o = a * w but quantized "

    # Convolution
    qaqw = convolve2d(a.quantized_values,w.quantized_values[::-1].T[::-1].T,mode='valid')

    # Scaling
    if out_scale is None:
        out_scale = 2 / (2**outBits - 1)
    newscale = a.scale * w.scale / out_scale
    m0, shift = convert_scale_to_shift_and_m0(
                    newscale,
                    precision=internalPrecision
                )

    fp_m = m0 * 2**(shift)

    # Reals accounting for quantized and fixed point error
    o_q = (qaqw * fp_m).astype(int)
    o_q = saturating_clip(o_q, outBits)

    # Quantized tensor of the output
    o_qtensor = quantized_tensor(
                    qaqw.shape,
                    outBits,
                    quantized_values=o_q,
                    scale=out_scale,
                    zero_point=0 # Assume 0 for now, might be bad later.
                )

    return o_qtensor
    
def convert_scale_to_shift_and_m0(scale,precision=16):
    " Convert scale to shift and zero point "
    shift = int(np.ceil(np.log2(scale)))
    m0 = scale / 2**shift
    fp_string = convert_to_fixed_point(m0,precision)
    m0_clipped = fixed_point_to_float(fp_string,precision)
    return m0_clipped, shift

def convert_to_fixed_point(number,precision):
    " Convert a float [0,1] to fixed point binary "
    out = ''
    for i in range(precision):
        number *= 2
        integer = int(number)
        number -= integer
        out += str(integer)
    return out

def fixed_point_to_float(number,precision):
    " Convert a fixed point binary to float [0,1] "
    out = 0
    for i in range(precision):
        out += int(number[i]) * 2**-(i+1)
    return out

def right_shift(number,shift):
    " Right shift a number. Shifting rounds toward -infty"
    return int(np.floor(number / 2.**(shift)))

def get_array_bits(array,signed=True):
    " Get the number of bits required to represent an array "
    return int(np.ceil(np.log2(np.abs(array).max())))+signed

def saturating_clip_old (num_i, inBits = 16, outBits = 8):
    '''
    Saturating clip.

    This only implemented saturation before. Now it also clips.
    '''

    # floor to round towards negative infinity
    num_i_shifted = right_shift(num_i, inBits - outBits)

    min = -(2**(outBits-1))
    max = 2**(outBits-1)-1
    
    if(num_i_shifted < min):
        return min
    if(num_i_shifted > max):
        return max
    
    return num_i_shifted

saturating_clip = np.vectorize(saturating_clip_old)

def binary_array_to_int(array,signed=False,outBits=None):
    ''' 
    Convert a binary array to an integer 
    The binary array is assumed to be in the last dimension
    '''
    out = 0
    arr = np.moveaxis(array,-1,0)
    for bit in arr:
        out = (out << 1) | bit
    return out

def int_to_bin(n, bits):
    " Convert an integer to a binary array "
    return np.moveaxis(np.array([n >> i & 1 for i in range(bits-1,-1,-1)]),0,-1)

def int_to_trit(n, trits):
    " Convert an integer to a trit (ternary) array "
    a = int_to_bin(abs(n), trits).T
    tritter = (((n < 0) * -2) + 1) # Bipolar array
    return (a * tritter)[1:]

def simulate_trit_mul(w,x, trits, verbose=False):
    " trit-decomposed multiplication "
    x_trits = int_to_trit(x, trits)
    partials = x_trits @ w.T

    print(x_trits)
    print(partials)

    return po2_accumulate(partials)

def po2_accumulate(array):
    " Accumulate an array of powers of 2 "
    ydim, xdim = array.shape
    po2 = 2**np.arange(0, ydim, 1)[::-1]
    po2 = po2.repeat(xdim).reshape(ydim,xdim)
    return (array * po2).sum(axis=0)