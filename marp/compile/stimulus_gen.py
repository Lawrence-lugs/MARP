import numpy as np 
import os
from ..quantization import quant
from ..onnx_tools import onnx_utils
from ..compile import compute
import onnx
import torch.nn.functional as F
import torch 
import rectpack

def _hex_but_no_0x(x):
    return hex(x)[2:]

def _hex3(n,hexits=8):
    return "%s"%("00000000%x"%(n&0xffffffff))[-hexits:]

hex_but_no_0x = np.vectorize(_hex3)
vhex3 = np.vectorize(_hex3)

def kernel_to_writes(kernel, channels, hexes=True):
    '''
    Packs every 4 elements of the kernel channel dimension (in HWC) into a single 32-bit integer
    Pads with zeros if the flattened kernel is not a multiple of 4.

    If hexes is True, returns a list of hex strings, otherwise returns a numpy array of integers.
    ''' 
    kernel_flat = kernel.reshape(-1, channels)
    kernel_flat = np.pad(kernel_flat, ((0, 0), (0, 4 - (kernel_flat.shape[1] % 4) if kernel_flat.shape[1] % 4 != 0 else 0)), mode='constant', constant_values=0)
    kernel_flat = kernel_flat.reshape(-1, 4)
    if hexes:
        kernel_packed = vhex3(kernel_flat, hexits=2)
        kernel_writes = []
    else:
        kernel_writes = np.zeros(kernel_flat.shape[0], dtype=np.uint32)
    for i in range(kernel_flat.shape[0]):
        if hexes:
            kernel_writes.append(f'{kernel_packed[i,3]}{kernel_packed[i,2]}{kernel_packed[i,1]}{kernel_packed[i,0]}')
        else:
            kernel_writes[i] += (kernel_flat[i,3] & 0xFF) << 24 | (kernel_flat[i,2] & 0xFF) << 16 | (kernel_flat[i,1] & 0xFF) << 8 | (kernel_flat[i,0] & 0xFF)
    return kernel_writes

def generate_random_torch_conv(
    ifmap_shape,
    ifmap_bits,
    kernel_shape,
    kernel_bits,
    stride,
    seed = 0
):
    
    t_ifmap = torch.randint(0, 2**(ifmap_bits), ifmap_shape, dtype=torch.int32)
    t_kern = torch.randint(0, 2**(kernel_bits), kernel_shape, dtype=torch.int32)
    t_res = F.conv2d(t_ifmap, t_kern, padding=1, stride=stride).numpy()

    t_matrix = t_kern.permute(0,2,3,1).numpy()
    t_matrix = t_matrix.reshape(t_kern.shape[0],-1).T  

    t_toeplitz = compute.toeplitzize_input(
        in_tensor=t_ifmap.squeeze(0).numpy(), 
        ksize=kernel_shape[-1], 
        strides=stride,
        channel_minor=True
    )

    a = t_toeplitz @ t_matrix
    output_shape = (
        ifmap_shape[0],
        kernel_shape[0],
        ifmap_shape[2]//stride, 
        ifmap_shape[3]//stride
    )
    out = a.T.reshape(*output_shape).astype(int)

    return t_res, t_matrix, t_ifmap, t_toeplitz, out

def generate_tensor(shape, mode = 'random',seed = 0):
    if mode == 'random':
        np.random.seed(seed)
        return np.random.rand(*shape)
    elif mode == 'linspace':
        return np.linspace(0, 1, num = np.prod(shape)).reshape(shape)

def _generate_random_quantized_acts(
    ifmap_shape,
    ifmap_bits,
    acts_mode
):
    
    if acts_mode not in ['counting','random']:
        raise ValueError(f"acts_mode must be 'counting' or 'random', got {acts_mode}")

    qa_proto = quant.QuantizedTensor(shape = ifmap_shape, precision = ifmap_bits, mode='3sigma')
    if acts_mode == 'counting':
        ifmap_size = np.prod(ifmap_shape)
        a_qvals = np.arange(0,ifmap_size, dtype=np.uint8) % 100
        a_qvals = a_qvals.reshape(ifmap_shape)
        qa = quant.QuantizedTensor(quantized_values = a_qvals, scale = qa_proto.scale, zero_point=0) 
    if acts_mode == 'random':
        qa = qa_proto   

    qa.quantized_values = qa.quantized_values.astype(np.uint8)
    return qa

def _generate_random_quantized_weights(
    kernel_shape,
    kernel_bits,
    kernel_dtype = np.int8,
    weight_density = 0.5,
    weight_mode = 'random'
):
    
    if kernel_bits == 1:
        if weight_mode == 'random':
            dist = np.random.rand(*kernel_shape)
            b = (dist < weight_density).astype(int)
        elif weight_mode == 'spatial':
            b = np.zeros(kernel_shape)
            for k in range(kernel_shape[0]):
                # b[k][0][2][2] = 1
                b[k][0][k % 3][k % 3] = 1
        scale = np.random.uniform(0.01,0.2,kernel_shape[0])
        qb = quant.QuantizedTensor(quantized_values = b, scale = scale, zero_point=0)
    else:
        qb = quant.QuantizedTensor(shape = kernel_shape, precision = kernel_bits, mode='maxmin')

    qb.quantized_values = qb.quantized_values.astype(kernel_dtype)

    return qb

def infer_ofmap_shape(
    ifmap_shape,
    kernel_shape,
    pads,
    stride
):
    '''
    Infer the output shape of a convolution (or matmul) operation.
    '''
    stridex = stride[0]
    stridey = stride[1]
    ofmap_shape = (
        ifmap_shape[0],
        kernel_shape[0],
        (ifmap_shape[2] - kernel_shape[2] + 2 * pads[0]) // stridex + 1,
        (ifmap_shape[3] - kernel_shape[3] + 2 * pads[2]) // stridey + 1
    )
    return ofmap_shape

def sample_onnx_qlinearconv(
    ifmap_shape,
    ifmap_bits,
    kernel_shape,
    kernel_bits,
    kernel_dtype,
    pads,
    stride,
    depthwise = False,
    weight_density = 0.5,
    seed = 0,
    acts_mode = 'random',
    weight_mode = 'random'
):
    
    '''
    Generates a sample QLinearConv
    * supports precisions < 8
    * only uint8 ifmap
    * acts_mode can be 'counting' or 'random'
    '''

    np.random.seed(seed)

    kernel_bits = 7 if depthwise else 1

    qa = _generate_random_quantized_acts(
        ifmap_shape=ifmap_shape,
        ifmap_bits=ifmap_bits,
        acts_mode=acts_mode
    )

    qb = _generate_random_quantized_weights(
        kernel_shape=kernel_shape,
        kernel_bits=kernel_bits,
        weight_density=weight_density,
        weight_mode=weight_mode,
        kernel_dtype=kernel_dtype
    )

    ofmap_shape = infer_ofmap_shape(
        ifmap_shape=ifmap_shape,
        kernel_shape=kernel_shape,
        pads=pads,
        stride=stride
    )
    sample_outs = quant.QuantizedTensor(shape = ofmap_shape, precision = 8, mode='3sigma')
    sample_outs.quantized_values = sample_outs.quantized_values.astype(np.uint8)

    scale_x = np.array(qa.scale, dtype=np.float32)
    zp_x = np.array(qa.zero_point, dtype=np.uint8)
    scale_w = np.array(qb.scale, dtype=np.float32)
    zp_w = np.array(qb.zero_point, dtype=kernel_dtype)
    scale_y = np.array(sample_outs.scale, dtype=np.float32)
    zp_y = np.array(sample_outs.zero_point, dtype=np.uint8)

    node_initializers = {
        # 'x':qa.quantized_values,
        'x_scale':scale_x,
        'x_zero_point':zp_x,
        'w':qb.quantized_values,
        'w_scale':scale_w,
        'w_zero_point':zp_w,
        'y_scale':scale_y,
        'y_zero_point':zp_y,
    }

    nx_node = onnx.helper.make_node(
        "QLinearConv",
        inputs=[
            "x",
            "x_scale",
            "x_zero_point",
            "w",
            "w_scale",
            "w_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        outputs=["y"],
        pads = pads,
        strides = stride,
        group = 1 if not depthwise else kernel_shape[0],
    )

    nx_model = onnx_utils.make_single_node_model(
        nx_node = nx_node,
        initializer_dict = node_initializers,
        input_names = ['x'],
        output_names = ['y']
    )
    ifmap = qa.quantized_values
    
    return nx_node, nx_model, ifmap

def map_single_matrix(matrix, core_shape, x_offset = 0, y_offset = 0, randomize = False):

    if randomize:
        matrix_map = np.random.randint(0, 2, core_shape, dtype=int)
    else:
        matrix_map = np.zeros(core_shape, dtype=int)
    matrix_map[y_offset:y_offset+matrix.shape[0], x_offset:x_offset+matrix.shape[1]] = matrix

    return matrix_map

def mapped_matrix_to_bank_writes(matrix_map, num_bank_cols = 32):

    matrix_map_banked = matrix_map.reshape(-1, num_bank_cols)
    write_array = quant.array_bin_to_int(matrix_map_banked.T[::-1].T)

    return write_array

def infer_optimal_adc_range_shifts(tplitz_act_vector, weights, ifmap_bits = 8):
    first_partials = quant.int_to_trit(tplitz_act_vector,ifmap_bits).T @ weights
    mean_partial = first_partials.mean()
    if mean_partial < 1:
        adc_ref_range_shifts = 0
    adc_ref_range_shifts = np.ceil(np.log(mean_partial)/np.log(2)) - 3
    if adc_ref_range_shifts < 0:
        adc_ref_range_shifts = 0
    if np.isnan(adc_ref_range_shifts):
        adc_ref_range_shifts = 0

    return adc_ref_range_shifts

def minorize_pad_ifmap(t_ifmap, padding=1, act_zero_point=0):
    """
    Prepare the input feature map for convolution by padding it.
    """
    t_ifmap = torch.from_numpy(t_ifmap)
    zp_x = float(act_zero_point)
    ifmap_channel_minor = F.pad(t_ifmap, (padding,padding,padding,padding), mode='constant', value=zp_x)    
    ifmap_channel_minor = ifmap_channel_minor.permute(0,2,3,1) # N C H W -> N H W C
    ifmap_channel_minor = ifmap_channel_minor.numpy()
    return ifmap_channel_minor

def pack_ifmap_to_ints(t_ifmap):
    """
    Pack the input feature map into integers.
    """
    ifmap_channel_packed_ints = [int(i,base=16) for i in quant.as_packed_hex(t_ifmap)]
    ifmap_channel_packed_ints = np.array(ifmap_channel_packed_ints)
    return ifmap_channel_packed_ints

def pad_scaler_writes(scale_to_map,core_shape,output_zero_point,mm_offset_x):
    if scale_to_map.ndim == 0:
        scale_to_map = np.repeat(scale_to_map, core_shape[1])
    scale = np.ones((core_shape[1],), dtype=np.float32)
    scales_width = scale_to_map.shape[0]
    scale[mm_offset_x:mm_offset_x+scales_width] = scale_to_map
    m0, shift = quant.vconvert_scale_to_shift_and_m0(scale, precision=16)
    int_scale = quant.vconvert_to_fixed_point_int(m0,16)
    scaler_data = output_zero_point * (2**20) + int_scale * (2**4) + (-shift)
    return scaler_data

def pad_bias_data(biases_to_map, core_shape, mm_offset_x):
    biases = np.zeros((core_shape[1],), dtype=np.int32)
    biases_width = biases_to_map.shape[0]
    biases[mm_offset_x:mm_offset_x+biases_width] = biases_to_map
    return biases

def imc_matrix_to_writes(t_matrix, core_shape, x_offset, y_offset, num_bank_cols = 32):
    matrix_map = map_single_matrix(t_matrix, core_shape, x_offset=x_offset, y_offset=y_offset)
    write_array = mapped_matrix_to_bank_writes(matrix_map,num_bank_cols)
    return write_array

def write_input_files(res_dict, savepath):
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        for key, value in res_dict.items():
            if type(value) is dict:
                continue
            if value.dtype in ['int32','float64']:
                np.savetxt(f'{savepath}/{key}.txt', value.flatten(), fmt='%d')
            else:
                np.savetxt(f'{savepath}/{key}.txt', value.flatten(), fmt='%s')
            np.savetxt(f'{savepath}/{key}_shape.txt', value.shape, fmt='%d')

def generate_qracc_words(
    savepath,
    stride,
    ifmap_shape,
    ifmap_bits,
    kernel_shape,
    kernel_bits,
    core_shape,
    depthwise = False,
    padding = 1,
    mm_offset_x = 0,
    mm_offset_y = 0,
    seed = 0,
    soft_padding = False
    ):

    print(f't_res, t_matrix, t_ifmap, t_toeplitz, scaler_params = sample_onnx_qlinearconv(ifmap_shape={ifmap_shape},ifmap_bits={ifmap_bits},kernel_shape={kernel_shape},kernel_bits={kernel_bits},kernel_dtype = np.int8,pads = (1,1,1,1),stride = {stride},seed = {seed}, depthwise = {depthwise})')
    t_res, t_matrix, t_ifmap, t_toeplitz, scaler_params = sample_onnx_qlinearconv(
        ifmap_shape=ifmap_shape,
        ifmap_bits=ifmap_bits,
        kernel_shape=kernel_shape,
        kernel_bits=kernel_bits,
        kernel_dtype = np.int8,
        pads = (padding,padding,padding,padding),
        stride = stride,
        seed = seed,
        depthwise = depthwise,
        # weight_density=0.05,
        # acts_mode='counting',
        # weight_mode='spatial'
    )

    if not depthwise:
        write_array = imc_matrix_to_writes(t_matrix, core_shape, x_offset=mm_offset_x, y_offset=mm_offset_y)
    else:
        kernel = t_matrix
        kernel_hwc = kernel.transpose(1,2,3,0)
        write_array = kernel_to_writes(kernel_hwc, 32, hexes=False)

    scaler_data = pad_scaler_writes(scaler_params['scale'], 
                                        core_shape=core_shape,
                                        output_zero_point=scaler_params['output_zp'],
                                        mm_offset_x=mm_offset_x)
    bias_data = pad_bias_data(scaler_params['gph_node'][0].biases, 
                                        core_shape=core_shape, 
                                        mm_offset_x=mm_offset_x)
    
    loaded_ifmap_padding = padding if soft_padding else 0 

    minorized_padded_ifmap = minorize_pad_ifmap(t_ifmap, padding=loaded_ifmap_padding ,act_zero_point = scaler_params['zp_x'])

    if soft_padding:
        print(f'[STIM_GEN] Using SOFT padding with {loaded_ifmap_padding} pixels. IFMAP shape: {minorized_padded_ifmap.shape}')
    else:
        print(f'[STIM_GEN] Using HARDWARE padding. IFMAP shape: {minorized_padded_ifmap.shape}')

    raw_data = {
        'ifmap':pack_ifmap_to_ints(minorized_padded_ifmap),
        'biases':bias_data,
        'scales':scaler_data,
        'weights':write_array,
    }
        
    res_dict = {
        'result': t_res,
        'toeplitz': t_toeplitz,
        'ifmap': minorized_padded_ifmap,
        'matrix_raw': t_matrix, 
        'scaler_data': scaler_data,
        'biases': bias_data,
        'scaler_params' : scaler_params # Can be removed later
    }
    write_input_files(res_dict, savepath)

    return raw_data, res_dict

def generate_top_inputs(
    savepath,
    stride,
    ifmap_shape,
    ifmap_bits,
    kernel_shape,
    kernel_bits,
    core_shape,
    depthwise = False,
    padding = 1,
    mm_offset_x = 0,
    mm_offset_y = 0,
    seed = 0
):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("[STIM_GEN] Generating Sample QLinearConv")
    print(f"t_res, t_matrix, t_ifmap, t_toeplitz, scaler_params = sample_onnx_qlinearconv(ifmap_shape={ifmap_shape},ifmap_bits={ifmap_bits},kernel_shape={kernel_shape},kernel_bits={kernel_bits},kernel_dtype = np.int8,pads = (1,1,1,1),stride = {stride},seed = {seed})")

    t_res, t_matrix, t_ifmap, t_toeplitz, scaler_params = sample_onnx_qlinearconv(
        ifmap_shape=ifmap_shape,
        ifmap_bits=ifmap_bits,
        kernel_shape=kernel_shape,
        kernel_bits=kernel_bits,
        kernel_dtype = np.int8,
        pads = (padding,padding,padding,padding),
        stride = stride,
        seed = seed,
        depthwise = depthwise,
        # weight_density=0.05,
        # acts_mode='counting',
        # weight_mode='spatial'
    )
    out = t_res

    print(f'matrix_map = map_single_matrix({t_matrix}, {core_shape}, x_offset={mm_offset_x}, y_offset={mm_offset_y})')
    matrix_map = map_single_matrix(t_matrix, core_shape, x_offset=mm_offset_x, y_offset=mm_offset_y)
    write_array = mapped_matrix_to_bank_writes(matrix_map,32)

    # Software padding and channel minor
    t_ifmap = torch.from_numpy(t_ifmap)
    zp_x = float(scaler_params['zp_x'])
    ifmap_channel_minor = F.pad(t_ifmap, (padding,padding,padding,padding), mode='constant', value=zp_x)
    ifmap_channel_minor = ifmap_channel_minor.permute(0,2,3,1) # N C H W -> N H W C
    ifmap_channel_minor = ifmap_channel_minor.numpy()

    # Press activations into 32b words of 4 elements each
    ifmap_channel_packed_ints = [int(i,base=16) for i in quant.as_packed_hex(ifmap_channel_minor)]
    ifmap_channel_packed_ints = np.array(ifmap_channel_packed_ints)

    # Generate scaler data and shifts
    scale_to_map = scaler_params['scale']

    if scale_to_map.ndim == 0:
        scale_to_map = np.repeat(scale_to_map, core_shape[1])
    
    scale = np.ones((core_shape[1],), dtype=np.float32)
    scale[mm_offset_x:mm_offset_x+scaler_params['gph_node'][0].kernel.shape[0]] = scale_to_map

    m0, shift = quant.vconvert_scale_to_shift_and_m0(scale, precision=16)
    int_scale = quant.vconvert_to_fixed_point_int(m0,16)
    scaler_data = scaler_params['output_zp'] * (2**20) + int_scale * (2**4) + (-shift)
    scaler_data = scaler_data

    # Prepare biases
    biases_to_map = scaler_params['gph_node'][0].biases
    biases = np.zeros((core_shape[1],), dtype=np.int32)
    biases[mm_offset_x:mm_offset_x+scaler_params['gph_node'][0].biases.shape[0]] = biases_to_map

    print('=== Weight Array ===')
    print(matrix_map)

    res_dict = {
        'result': t_res,
        'toeplitz': t_toeplitz,
        'ifmap': ifmap_channel_packed_ints,
        'ifmap_ints': ifmap_channel_minor,
        'small_matrix': t_matrix,
        'matrix': write_array,
        'weights_np': matrix_map, 
        'scaler_data': scaler_data,
        'biases': biases,
    }

    if savepath is not None:
        for key, value in res_dict.items():
            if value.dtype in ['int32','float64']:
                np.savetxt(f'{savepath}/{key}.txt', value.flatten(), fmt='%d')
            else:
                np.savetxt(f'{savepath}/{key}.txt', value.flatten(), fmt='%s')
            np.savetxt(f'{savepath}/{key}_shape.txt', value.shape, fmt='%d')

    return res_dict


def generate_qracc_inputs(
    wDimX = 8,
    wDimY = 128,
    xBatches = 50,
    xTrits = 4,
    outBits = 4,
    seed = 0,
    weight_mode = 'bipolar',
    col_symmetric = False,
    rangeBits = None,
    x_repeat = False,
    clip_output = True,
    unsigned_acts = False,
    bitRange = None
):
    '''
    Generates integer weights and inputs, including clipped integer reference outputs.

    Parameters:
    wDimX : int -- Number of columns
    wDimY : int -- Number of rows
    xBatches : int -- Number of batches
    xTrits : int -- Number of trits for input data
    outBits : int -- Number of bits for output data
    seed : int -- Random seed
    weight_mode : str -- 'binary' or 'bipolar'
    col_symmetric : bool -- If True, all columns will have the same output value
    '''
    np.random.seed(seed)
    wShape = (wDimX, wDimY)
    xShape = (xBatches, wDimY)

    if col_symmetric:
        # In this mode, every column will have the same output value
        print('[STIM_GEN] Generating symmetric columns')
        if weight_mode=='binary':
            w = np.random.randint(0,2, wDimY)
            w = w.repeat(wDimX).reshape(wDimY, wDimX).T
        elif weight_mode=='bipolar':
            w = np.random.randint(0,2, wShape)*2-1
            w = w.repeat(wDimX).reshape(wDimY, wDimX).T
        else:
            raise ValueError('Invalid weight_mode')
    else:
        print('[STIM_GEN] Generating random weights')
        if weight_mode=='binary':
            # When doing unsigned acts with 1b binary weights, the output tends to oversaturate. We need to make the weights sparse to have testable outputs.
            w = np.random.randint(0,2, wShape)
            if unsigned_acts:
                a = np.random.rand(*wShape)
                w = (a < 0.1).astype(int) # Sparse Weights
        elif weight_mode=='bipolar':
            w = np.random.randint(0,2, wShape)*2-1
        else:
            raise ValueError('Invalid weight_mode')

    if bitRange is None:
        bitRange = xTrits
    if not unsigned_acts:
        low_lim = -(2**(bitRange)-1)
        high_lim = 2**(bitRange)
    else:
        low_lim = 0
        high_lim = 2**(bitRange)

    if x_repeat:
        x = np.random.randint(low_lim, high_lim,wDimY)
        x = x.repeat(xBatches).reshape(wDimY, xBatches).T
    else:
        x = np.random.randint(low_lim, high_lim,xShape)
    wx = w @ x.T

    # Get_array_bits is incorrect. The real bit count depends on the range of the input data
    # wxBits = quant.get_array_bits(wx)

    if rangeBits is None:
        wxBits = np.log2(wDimY) + 1 # +1 because of bipolar rep (-128,128) vs (0,128)
    else:
        wxBits = rangeBits

    if clip_output:
        wx_outBits = quant.saturating_clip(wx, inBits = wxBits, outBits = outBits)
    else:
        wx_outBits = wx
    return w, x, wx_outBits.T