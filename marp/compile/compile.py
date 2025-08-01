import numpy as np
from .stimulus_gen import *
import onnx
from ..onnx_tools import onnx_utils
from ..mapping import core
from ..compile import compute

class QrAccNodeCode(object):
    '''
    Holds all relevant information for testing a single node in QRAcc.
    Mappings and reference output attributes are NCHW
    '''

    def __init__(self, mapped_node : core.MappedNode, mapped_bin : core.MappedBin, ifmap, imc_core_size = (256,256), ws_core_size = 32, ifmap_bits = 8, ofmap_bits = 8, nx_model : onnx.ModelProto = None):

        self.ifmap = ifmap if ifmap.ndim == 4 else ifmap.reshape((1, -1, 1, 1))

        self.ofmap_shape = infer_ofmap_shape(
            ifmap_shape=self.ifmap.shape,
            kernel_shape=mapped_node.kernel.shape,
            pads=mapped_node.pads,
            stride=mapped_node.strides,
        )

        self.ifmap_bits = ifmap_bits
        self.ofmap_bits = ofmap_bits

        self.ws_core_size = ws_core_size
        self.imc_core_size = imc_core_size

        self.mapped_bin = mapped_bin
        self.mapped_node = mapped_node

        # if nx_model is None:
        if 0:
            self.reference_output = self._generate_reference_output()[0].transpose((0, 2, 3, 1))  # Convert to NHWC format
        else:
            output_tensor = mapped_node.nx_node.output[0]
            input_dict = self._get_input_dict()
            onnx_out = onnx_utils.get_intermediate_tensor_value(nx_model,output_tensor,input_dict=input_dict)

            # We pretend the output is in NCHW if it's a matmul (we treat those like pointwise convs)
            if len(onnx_out.shape) == 1:
                onnx_out = onnx_out.reshape((1, -1, 1, 1))
            self.reference_output = onnx_out.transpose((0, 2, 3, 1))
        
        self.toeplitz = compute.toeplitzize_input(
            in_tensor = self.ifmap.squeeze(axis=0), # Remove batch dimension for toeplitz
            kernel_shape=mapped_node.kernel.shape,
            strides=mapped_node.strides,
            pads=mapped_node.pads,
            zero_point=mapped_node.x_zp,
            channel_minor=True
        )

        self.nx_model = nx_model

        # The writable matrix is the kernel for depthwise nodes, because they go into qracc
        self.matrix = mapped_node.matrix if not mapped_node.depthwise else mapped_node.kernel

        self.inputs = self.mapped_node.get_true_inputs()  # Non-initializer inputs of the node
        self.outputs = self.mapped_node.nx_node.output

        return
    
    def _get_input_dict(self):
        if self.mapped_node.type == 'QLinearMatMul':
            in_tensor = self.ifmap.squeeze()  # Convert to a 1D tensor for matmul
        else:
            in_tensor = self.ifmap
        input_dict = {
            self.mapped_node.nx_node.input[0]: in_tensor,
        }
        return input_dict

    
    def config(self):
        '''
        Returns the configuration for the QRAcc node.
        '''

        ifmap_shape = self.ifmap.shape
        mapped_node = self.mapped_node
        ifmap_bits = self.ifmap_bits
        ofmap_bits = self.ofmap_bits

        if not mapped_node.depthwise:
            if ifmap_shape[1] != mapped_node.kernel.shape[1]:
                raise ValueError(f"Input feature map channels {ifmap_shape} do not match kernel input channels {mapped_node.kernel.shape}.")

        kernel_shape = mapped_node.kernel.shape
        # ofmap_dimx = ((ifmap_shape[2] - kernel_shape[2] + 2*mapped_node.pads[0]) // mapped_node.strides[0]) + 1 #(W-K+2P)/S + 1
        # ofmap_dimy = ((ifmap_shape[3] - kernel_shape[3] + 2*mapped_node.pads[1]) // mapped_node.strides[1]) + 1

        ofmap_dimx = self.ofmap_shape[2]
        ofmap_dimy = self.ofmap_shape[3]


        adc_ref_range_shifts = infer_optimal_adc_range_shifts(
            tplitz_act_vector=self.toeplitz[0],
            weights=mapped_node.matrix,
            ifmap_bits=ifmap_bits
        ) if not mapped_node.depthwise else 0

        config_dict = {
            "n_input_bits_cfg": ifmap_bits,
            "n_output_bits_cfg": ofmap_bits,
            "unsigned_acts": 1,
            "binary_cfg": 1,
            "adc_ref_range_shifts": int(adc_ref_range_shifts),
            "filter_size_y": kernel_shape[3],
            "filter_size_x": kernel_shape[2],
            "input_fmap_dimx": ifmap_shape[2],
            "input_fmap_dimy": ifmap_shape[3],
            "output_fmap_dimx": ofmap_dimx,
            "output_fmap_dimy": ofmap_dimy,
            "stride_x": mapped_node.strides[0],
            "stride_y": mapped_node.strides[1],
            "num_input_channels": ifmap_shape[1],
            "num_output_channels": kernel_shape[0],
            "mapped_matrix_offset_x": mapped_node.offset_x,
            "mapped_matrix_offset_y": mapped_node.offset_y,
            "padding": mapped_node.pads[0],  # Use 0 padding for soft padding
            "padding_value": mapped_node.x_zp,  # Padding value for the input feature map
        }

        return config_dict
    
    @classmethod
    def produce_single_node_test(
        cls,
        ifmap_shape,
        kernel_shape,
        pads,
        stride,
        depthwise,
        offset_x,
        offset_y,
        core_size,
        ws_core_size,
        ifmap_bits = 8,
        kernel_bits = 1
    ):
        
        nx_node, nx_model, ifmap = sample_onnx_qlinearconv(ifmap_shape=ifmap_shape,ifmap_bits=ifmap_bits,kernel_shape=kernel_shape,kernel_bits=kernel_bits,kernel_dtype = np.int8,pads = pads,stride = stride, seed = 0, depthwise = depthwise)

        mapped_node = core.MappedNode(nx_node, bin_id = 0, mapped_rect=None, nx_model = nx_model, offset_x=offset_x, offset_y = offset_y)

        bin_weights = np.zeros(core_size, dtype=np.int8)
        bin_weights[offset_y:offset_y+mapped_node.matrix.shape[0], offset_x:offset_x+mapped_node.matrix.shape[1]] = mapped_node.matrix

        mapped_bin = core.MappedBin(
            bin_id = 0,
            weights = bin_weights
        )

        u_code = QrAccNodeCode(
            mapped_node = mapped_node,
            mapped_bin = mapped_bin,
            ifmap = ifmap,
            imc_core_size = core_size,
            ws_core_size = ws_core_size,
            nx_model = nx_model,
        )

        return u_code
    
    def _get_weight_data(self):
        if self.mapped_node.depthwise:

            if self.mapped_node.kernel.shape[0] < self.ws_core_size:
                physical_kernel = np.zeros((self.ws_core_size, *self.mapped_node.kernel.shape[1:]), dtype=self.mapped_node.kernel.dtype)
                physical_kernel[:self.mapped_node.kernel.shape[0]] = self.mapped_node.kernel
            else:
                physical_kernel = self.mapped_node.kernel

            physical_kernel_hwc = physical_kernel.transpose(1,2,3,0)  # Convert to KHWC format
            return kernel_to_writes(physical_kernel_hwc, channels=self.ws_core_size, hexes=False)
        else:
            return mapped_matrix_to_bank_writes(self.mapped_bin.weights,32)
        
    def _get_scaler_data(self):
        return pad_scaler_writes(
            scale_to_map = self.mapped_node.scale,
            core_shape= self.imc_core_size,
            output_zero_point = self.mapped_node.y_zp,
            mm_offset_x= self.mapped_node.offset_x,
        )
    
    def _get_bias_data(self):
        return pad_bias_data(self.mapped_node.biases, 
            core_shape=self.imc_core_size, 
            mm_offset_x=self.mapped_node.offset_x,
        )
    
    def _get_ifmap_data(self):
        ifmap_hwc = self.ifmap.transpose(0,2,3,1)
        return pack_ifmap_to_ints(ifmap_hwc)

    def write_files(self, savepath):
        '''
        Write reference text files for the QRAcc RTL testbench.
        Optional. Only used to debug.
        '''
        
        res_dict = {
            'result': self.reference_output,
            'toeplitz': self.toeplitz,
            'ifmap': self.ifmap.transpose(0,2,3,1),  # Convert to NHWC format
            'matrix_raw': self.matrix,  
            'scaler_data': self._get_scaler_data(),
            'biases': self._get_bias_data()
        }

        write_input_files(res_dict, savepath)

    def _generate_reference_output(self):
        node_initializers = [
            self.ifmap,
            self.mapped_node.x_scale,
            self.mapped_node.x_zp,
            self.mapped_node.kernel,
            self.mapped_node.w_scale,
            self.mapped_node.w_zp,
            self.mapped_node.y_scale,
            self.mapped_node.y_zp,
            self.mapped_node.biases
        ]
        sample_outs = np.random.randint(
            low=0, high=2**self.ofmap_bits, size=self.ofmap_shape, dtype=np.uint8
        )
        return onnx_utils.infer_node_output(
            self.mapped_node.nx_node,
            inputs=node_initializers,
            outputs=[sample_outs],
            name=self.mapped_node.name,
        )        
    
    def compile(self, include_ifmap_writes=True, write_weights=True, add_read=True, config_write_address='00000010', end=True, preserve_ifmap=False):
        '''
        Compile the node into a list of assembly instructions for QRAcc.
        include_ifmap_writes: bool, whether to include ifmap writes in the output.
        solo: bool, whether to compile the node as a standalone unit (default True).
        '''
        # print(f"Compiling node {self.mapped_node.node_id}:{self.mapped_node.name} for QRAcc...")

        config_dict = self.config()
        config_writes = bundle_config_into_write(config_dict, config_write_address)
        commands = config_writes
        
        if write_weights:
            if self.mapped_node.depthwise:
                commands += make_trigger_write('TRIGGER_LOADWEIGHTS_DIGITAL', write_address=config_write_address)
            else:
                commands += make_trigger_write('TRIGGER_LOADWEIGHTS', write_address=config_write_address)
            commands += write_array_to_asm(self._get_weight_data()) 
        
        # Writing to the scaler is not optional
        commands += make_trigger_write('TRIGGER_LOAD_SCALER', write_address=config_write_address)
        commands += write_array_to_asm(self._get_scaler_data()) 
        commands += write_array_to_asm(self._get_bias_data()) 
        
        if include_ifmap_writes: # If not, the ifmap is assumed to be already in the ACTMEM
            commands += make_trigger_write('TRIGGER_LOAD_ACTIVATION', write_address=config_write_address)
        commands += write_array_to_asm(self._get_ifmap_data())

        if self.mapped_node.depthwise:
            commands += make_trigger_write('TRIGGER_COMPUTE_DIGITAL', write_address=config_write_address, preserve_ifmap=preserve_ifmap)
        else:
            commands += make_trigger_write('TRIGGER_COMPUTE_ANALOG', write_address=config_write_address, preserve_ifmap=preserve_ifmap)
        commands += [f'WAITBUSY']
        
        if add_read:
            commands += make_trigger_write('TRIGGER_READ_ACTIVATION', write_address=config_write_address)
            commands += [f'WAITREAD']
            if end:
                commands += ['END']

        return commands
    
    def __repr__(self):
        # Print attributes of the class, showing shapes for arrays
        attrs = vars(self)
        attr_strs = []
        for key, value in attrs.items():
            if isinstance(value, np.ndarray):
                attr_strs.append(f"{key}=array(shape={value.shape})")
            elif key == 'mapped_node':
                attr_strs.append(f"{key}={value.name} (id={value.node_id})")
            elif key == 'mapped_bin' and value is not None:
                attr_strs.append(f"{key}={value.bin_id} (shape={value.weights.shape})")
            elif key == 'nx_model':
                attr_strs.append(f"{key}={value.graph.name} (nodes={len(value.graph.node)})")
            else:
                attr_strs.append(f"{key}={value}")
        attr_list = ',\n\t'.join(attr_strs)
        return f"QrAccNodeCode(\n\t{attr_list})"

def generate_random_intermediate_tensor(
    nx_model : onnx.ModelProto,
    node_id : int,
    input_dict : dict
):
    input_tensor_shape = onnx_utils.get_intermediate_tensor_value(
        nx_model, nx_model.graph.node[node_id].input[0], input_dict=input_dict).shape
    return np.random.randint(0, 256, input_tensor_shape).astype(np.uint8)

def is_nx_node_compilable(
    nx_node : onnx.NodeProto
):
    """
    Check if an ONNX node is compilable for QRAcc.
    Currently, only QLinearConv and QLinearMatMul nodes are supported.
    """
    if nx_node is None:
        return False
    if nx_node.op_type == 'QLinearConv':
        return True
    elif nx_node.op_type == 'QLinearMatMul':
        return True
    else:
        return False

def make_trigger_write(
    command,
    clear=0,
    inst_write_mode=0,
    csr_main_trigger_enum=None,
    write_address='00000010',  # Default CSR base address for MAIN
    preserve_ifmap=False  # Used for successive nodes that use the same ifmap
):
    """
    command: string, one of possible triggers in qracc_pkg.svh
    clear: 0 or 1, sets the clear bit
    inst_write_mode: 0 or 1, sets the inst_write_mode bit
    csr_main_trigger_enum: optional dict mapping string to value, otherwise uses default mapping
    write_address: hex string, address to write to (default 0x10)
    """
    # Default mapping from qracc_pkg.svh
    if csr_main_trigger_enum is None:
        csr_main_trigger_enum = {
            'TRIGGER_IDLE': 0,
            'TRIGGER_LOAD_ACTIVATION': 1,
            'TRIGGER_LOADWEIGHTS': 2,
            'TRIGGER_COMPUTE_ANALOG': 3,
            'TRIGGER_COMPUTE_DIGITAL': 4,
            'TRIGGER_READ_ACTIVATION': 5,
            'TRIGGER_LOADWEIGHTS_DIGITAL': 6,
            'TRIGGER_LOAD_SCALER': 7
        }
    if command not in csr_main_trigger_enum:
        raise ValueError(f"Unknown command: {command}")

    trigger_val = csr_main_trigger_enum[command] & 0x7
    clear_val = (clear & 0x1) << 3
    inst_write_mode_val = (inst_write_mode & 0x1) << 5
    preserve_ifmap_val = (preserve_ifmap & 0x1) << 12  # Preserve ifmap bit

    word = trigger_val | clear_val | inst_write_mode_val | preserve_ifmap_val
    return [f'LOAD {write_address} {word:08x}']

def bundle_config_into_write(
    config_dict,
    config_write_address = '00000010' # in hex, base address for CSR
):
    # CSR register addresses (offsets from base)
    CSR_REG_CONFIG      = 1
    CSR_REG_IFMAP_DIMS  = 2
    CSR_REG_OFMAP_DIMS  = 3
    CSR_REG_CHANNELS    = 4
    CSR_REG_OFFSETS     = 5
    CSR_REG_PADDING     = 6

    base_addr = int(config_write_address, 16)

    # Pack fields into 32-bit words
    config_word = (
        ((config_dict['n_output_bits_cfg']      & 0xF) << 28) |
        ((config_dict['n_input_bits_cfg']       & 0xF) << 24) |
        ((config_dict['stride_y']               & 0xF) << 20) |
        ((config_dict['stride_x']               & 0xF) << 16) |
        ((config_dict['filter_size_x']          & 0xF) << 12) |
        ((config_dict['filter_size_y']          & 0xF) << 8)  |
        ((config_dict['adc_ref_range_shifts']   & 0xF) << 4)  |
        ((config_dict['unsigned_acts']          & 0x1) << 1)  |
        ((config_dict['binary_cfg']             & 0x1) << 0)
    )
    ifmap_dims_word = (
        ((config_dict['input_fmap_dimy'] & 0xFFFF) << 16) |
        ((config_dict['input_fmap_dimx'] & 0xFFFF) << 0)
    )
    ofmap_dims_word = (
        ((config_dict['output_fmap_dimy'] & 0xFFFF) << 16) |
        ((config_dict['output_fmap_dimx'] & 0xFFFF) << 0)
    )
    channels_word = (
        ((config_dict['num_output_channels'] & 0xFFFF) << 16) |
        ((config_dict['num_input_channels']  & 0xFFFF) << 0)
    )
    offsets_word = (
        ((config_dict['mapped_matrix_offset_y'] & 0xFFFF) << 16) |
        ((config_dict['mapped_matrix_offset_x'] & 0xFFFF) << 0)
    )
    padding_word = (
        ((config_dict['padding_value'] & 0xFFFF) << 4) |
        ((config_dict['padding'] & 0xFFFF) << 0)
    )

    # List of (address, hex_word) for each CSR
    config_writes = [
        f"LOAD {base_addr + CSR_REG_CONFIG:08x} {config_word:08x}",
        f"LOAD {base_addr + CSR_REG_IFMAP_DIMS:08x} {ifmap_dims_word:08x}",
        f"LOAD {base_addr + CSR_REG_OFMAP_DIMS:08x} {ofmap_dims_word:08x}",
        f"LOAD {base_addr + CSR_REG_CHANNELS:08x} {channels_word:08x}",
        f"LOAD {base_addr + CSR_REG_OFFSETS:08x} {offsets_word:08x}",
        f"LOAD {base_addr + CSR_REG_PADDING:08x} {padding_word:08x}"  # Assuming padding is at CSR_REG_PADDING
    ]
    
    return config_writes

def write_array_to_asm(write_array, address='00000100'):
    asm = []
    for element in write_array:
        asm.append(f"LOAD {address} {vhex3(element)}")
    return asm

def get_config_from_mapped_node(
    mapped_node : core.MappedNode,
    ifmap_shape : tuple,
    ifmap_bits = 8,
    ofmap_bits = 8
):
    '''
    Generate QRAcc configuration for a QLinearConv node.
    Needs information about the previous ifmap shape.
    '''

    if ifmap_shape[1] != mapped_node.kernel.shape[1]:
        raise ValueError("Input feature map channels do not match kernel input channels.")

    kernel_shape = mapped_node.kernel.shape
    ofmap_dimx = ((ifmap_shape[2] - kernel_shape[2] + 2*mapped_node.pads[0]) // mapped_node.strides[0]) + 1 #(W-K+2P)/S + 1
    ofmap_dimy = ((ifmap_shape[3] - kernel_shape[3] + 2*mapped_node.pads[1]) // mapped_node.strides[1]) + 1

    tplitz_window_length = kernel_shape[2] * kernel_shape[3] * ifmap_shape[1] # CFxFy

    tplitz_act_vector = np.random.randint(
        low=0, high=2**ifmap_bits, size=tplitz_window_length)

    adc_ref_range_shifts = infer_optimal_adc_range_shifts(
        tplitz_act_vector=tplitz_act_vector,
        weights=mapped_node.matrix.T,
        ifmap_bits=ifmap_bits
    )

    config_dict = {
        "n_input_bits_cfg": ifmap_bits,
        "n_output_bits_cfg": ofmap_bits,
        "unsigned_acts": 1,
        "binary_cfg": 1,
        "adc_ref_range_shifts": int(adc_ref_range_shifts),
        "filter_size_y": kernel_shape[3],
        "filter_size_x": kernel_shape[2],
        "input_fmap_dimx": ifmap_shape[2],
        "input_fmap_dimy": ifmap_shape[3],
        "output_fmap_dimx": ofmap_dimx,
        "output_fmap_dimy": ofmap_dimy,
        "stride_x": mapped_node.strides[0],
        "stride_y": mapped_node.strides[1],
        "num_input_channels": ifmap_shape[1],
        "num_output_channels": kernel_shape[0],
        "mapped_matrix_offset_x": mapped_node.offset_x,
        "mapped_matrix_offset_y": mapped_node.offset_y,
        "padding": mapped_node.pads[0],  # Use 0 padding for soft padding
        "padding_value": mapped_node.x_zp,  # Padding value for the input feature map
    }

    return config_dict

def check_if_any_consumers_are_noncompilable(
    nx_model : onnx.ModelProto,
    nx_node : int
):
    """
    Check if any consumers of the given node are non-compilable.
    Returns True if any consumer is non-compilable, False otherwise.
    """
    for output in nx_node.output:
        for node in nx_model.graph.node:
            if output in node.input and not is_nx_node_compilable(node):
                return True
    return False

def traverse_and_compile_nx_graph(
    nx_model     : onnx.ModelProto,
    input_dict   : dict,
    imc_core_size: tuple = (256, 256),
    dwc_core_size: int = 32,
    until        : int = None,
    starting     : int = 0,
    packer = None
):
    
    u_nx_mapping = core.NxModelMapping(
        nx_model,
        imc_core_size=imc_core_size,
        dwc_core_size=dwc_core_size,
        packer = packer
    )

    if until is None:
        until = len(nx_model.graph.node)
    commands = []

    prev_bin_id = None
    current_loaded_ifmap_name = None
    for node_id,nx_node in enumerate(nx_model.graph.node[starting:until]):
        
        readout = check_if_any_consumers_are_noncompilable(nx_model, nx_node)
        next_node = nx_model.graph.node[node_id + 1] if node_id + 1 < len(nx_model.graph.node) else None

        if is_nx_node_compilable(nx_node):
            input_tensor = generate_random_intermediate_tensor(
                nx_model, node_id, input_dict
            )
            u_code = QrAccNodeCode(
                mapped_node   = u_nx_mapping.get_mapped_node_by_id(node_id),
                mapped_bin    = u_nx_mapping.get_bin_of_node_id(node_id),
                ifmap         = input_tensor,
                imc_core_size = imc_core_size,
                ws_core_size  = dwc_core_size,
                nx_model      = nx_model
            )

            preserve_ifmap = False
            if is_nx_node_compilable(next_node) and next_node.input[0] == u_code.inputs[0]:
                preserve_ifmap = True

            commands += ['INFO']
            commands += [f'{sanitize_name(nx_node.name)}'] # The tb will later parse this as the node name
            commands += [f'Current loaded ifmap: {current_loaded_ifmap_name}']

            print('============ Compiling Node ============')

            if u_code.mapped_node.depthwise:
                print(f"Compiling {nx_node.name} as depthwise node...")
            
            if (u_code.inputs[0] != current_loaded_ifmap_name):
                print(f"Compiling {nx_node.name} as first node of set...")
                commands += [f'NODE {nx_node.name} (id={u_code.mapped_node.node_id}) reading ifmap ({u_code.inputs[0]}) from external memory']
            else:
                print(f"Compiling {nx_node.name} as middle node of set...")
                commands += [f'NODE {nx_node.name} (id={u_code.mapped_node.node_id}) will read ifmap ({u_code.inputs[0]}) from ACTMEM']

            if readout:
                print(f"{nx_node.name} is followed by a non-compilable node, so it will write the ofmap to external memory.")
                commands += [f'NODE {nx_node.name} (id={u_code.mapped_node.node_id}) will write ofmap ({u_code.outputs[0]}) into external memory']

            if u_code.mapped_node.bin_id is not None:
                if u_code.mapped_node.bin_id != prev_bin_id:
                    print(f"Rewriting bin {nx_node.name} as bin changed from {prev_bin_id} to {u_code.mapped_node.bin_id}...")
                    commands += [f'NODE {nx_node.name} (id={u_code.mapped_node.node_id}) will rewrite the bin from {prev_bin_id} to {u_code.mapped_node.bin_id}']
                else:
                    print(f'NODE {nx_node.name} (id={u_code.mapped_node.node_id}) BIN COMBO!!! Reusing bin {u_code.mapped_node.bin_id}.')
                    commands += [f'This node will reuse the bin {u_code.mapped_node.bin_id} from the previous node']
            else:
                commands += [f'NODE {nx_node.name} (id={u_code.mapped_node.node_id}) is a depthwise node. Loaded bin ({prev_bin_id}) is preserved.']

            if preserve_ifmap:
                commands += [f'NODE {nx_node.name} (id={u_code.mapped_node.node_id}) will preserve the ifmap ({u_code.inputs[0]}) in ACTMEM for the next node.']
                    
            # commands += [u_code.__repr__()]
            commands += ['ENDINFO']
            commands += u_code.compile(
                include_ifmap_writes = (u_code.inputs[0] != current_loaded_ifmap_name) ,
                write_weights        = u_code.mapped_node.bin_id                       != prev_bin_id,
                add_read             = readout,
                end                  = False ,
                preserve_ifmap       = preserve_ifmap ,
            )

            prev_bin_id = u_code.mapped_node.bin_id if not u_code.mapped_node.depthwise else prev_bin_id  # If the node is depthwise, we don't change the bin id, as it will be the same as the previous node
        
            if not preserve_ifmap:
                current_loaded_ifmap_name = u_code.outputs[0]

        else:
            print(f'Skipping {nx_node.name} as it is not compilable...')

    commands += ['END'] 
    prev_node = nx_node

    return commands

def get_info_command(u_code):

    """
    Generate an INFO command for the QRAcc testbench.
    This command is used to print information about the node.
    """
    commands = []
    commands += ['INFO']
    commands += [u_code.__repr__()]
    commands += ['ENDINFO']
    return commands

def sanitize_name(name):
    """
    Sanitize a node name to be used as filenames in the QRAcc testbench.
    Removes special characters and directory traversal attempts.
    Returns a safe string that can be used as a filename.
    """
    # Replace any chars that could be problematic in filenames
    safe_chars = name.replace(' ', '_')
    safe_chars = ''.join(c for c in safe_chars if c.isalnum() or c in '_-')
    
    # Prevent directory traversal attempts
    safe_name = safe_chars.replace('..', '')
    
    # Avoid empty names
    if not safe_name:
        safe_name = 'node'
        
    return safe_name