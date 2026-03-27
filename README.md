# MARP

**Mapping Accelerator for Reconfigurable Packing** — rectangular bin-packing algorithms that pack multiple DNN layers into a single weight matrix written onto Analog In-Memory Computing (AIMC) arrays.  MARP reduces the number of AIMC cores needed to fully map models **or** reduces the number of weight writes during a full-model inference.

MARP is a subset of the functionalities in the [hardware accelerator design garage](https://github.com/Lawrence-lugs/hwacc_design_garage).

## Installation

```bash
# Clone (with ZigZag submodule)
git clone --recurse-submodules https://github.com/Lawrence-lugs/MARP.git
cd MARP

# Install ZigZag from submodule, then MARP itself
pip install -e third_party/zigzag
pip install -e ".[dev]"
```

> **Dev Container:** Opening this repo in VS Code with the Dev Containers extension will build and configure everything automatically.

## Quick start

```python
import onnx
from marp import NxModelMapping, QRAccModel, get_packer_by_type

nx_model = onnx.load("onnx_models/ad_quantized_int8.onnx")
packer = get_packer_by_type("Dense")
mapping = NxModelMapping(nx_model, imc_core_size=(256, 256), packer=packer)
model = QRAccModel(mapping, num_cores=1)

print(f"Utilisation: {model.utilization:.2%}")
print(f"Weight writes: {model.weight_bin_writes}")
```

### ZigZag analytical modelling

```python
from marp.accelerator.zigzag_bridge import estimate_hardware_performance

result = estimate_hardware_performance(
    workload="onnx_models/ad_quantized_int8.onnx",
    accelerator="accelerator_configs/aimc_example.yaml",
    mapping="accelerator_configs/mapping_example.yaml",
)
print(f"Energy: {result['energy']:.2e} pJ  |  Latency: {result['latency']:.0f} cycles")
```

### Key Components

- **`marp/`** — Core library: compilation, mapping, ONNX processing, quantisation, ZigZag bridge
- **`marp_results.ipynb`** — Interactive notebook reproducing the paper's experiments
- **`onnx_models/`** — Quantised models (AD, IC, KS, MBV2)
- **`third_party/zigzag/`** — MICAS ZigZag submodule for analytical accelerator modelling
- **`tests/`** — pytest suite (4 models × 4 packers × 3 tests)

## Testing

```bash
pytest                 # run all tests
pytest -k "test_plot"  # run only plotting tests
```

### Test Coverage

1. MARP packing with 4 models × 4 packer types (Naive, Dense, Balanced, WriteOptimized)
2. MARP compilation of 3 models × 4 packer types

## Reproducing results

To reproduce the reported results, proceed to `marp_results.ipynb`.

## Project Structure

```
MARP/
├── pyproject.toml              # Package definition & dependencies
├── LICENSE                     # MIT
├── README.md
├── marp_results.ipynb          # Results notebook
├── onnx_models/                # Quantised ONNX models
│
├── marp/                       # Core library (pip-installable)
│   ├── __init__.py             # Public API re-exports
│   ├── constants.py            # Hardware constants & Trigger enum
│   ├── accelerator/            # ZigZag integration layer
│   │   └── zigzag_bridge.py
│   ├── compile/                # Assembly generation for QRAcc
│   │   ├── compile.py
│   │   ├── compute.py
│   │   └── stimulus_gen.py
│   ├── mapping/                # Bin packing & layer→core mapping
│   │   ├── core.py
│   │   └── packer_utils.py
│   ├── onnx_tools/             # ONNX graph surgery
│   │   ├── onnx_splitter.py
│   │   └── onnx_utils.py
│   └── quantization/
│       └── quant.py
│
├── third_party/
│   └── zigzag/                 # Git submodule (MICAS ZigZag v3)
│
└── tests/
└── tests/
    ├── conftest.py             # Fixtures & MODEL_DIR
    └── test_marp.py            # Parametrised tests
```
