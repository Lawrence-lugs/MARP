# MARP

We build MARP on the idea of using rectangular bin ### Key Components

- **`marp/`**: The core library implementing the MARP algorithm with modules for compilation, mapping, ONNX processing, and quantization
- **`marp_results.ipynb`**: Interactive notebook containing all experiments and results from the paper
- **`onnx_models/`**: Collection of quantized models used in experiments (AD: Anomaly Detection, IC: Image Classification, KS: Keyword Spotting, MBV2: MobileNetV2)
- **`images/`**: Visualization results showing different packing strategies and performance comparisons
- **`tests/`**: Comprehensive test suite for validating MARP functionality

## Testing with pytest

MARP uses pytest for comprehensive testing to ensure the reliability and correctness of all core components.

### Test Structure

The test suite is organized as follows:

- **`tests/test_marp.py`**: Main test module covering core MARP functionality

### Test Coverage

The test suite covers the following key areas:

1. **Model Loading and Processing**: Tests for ONNX model loading, quantization, and preprocessing
2. **Packing Algorithms**: Comprehensive testing of all four packing strategies:
   - Naive packing
   - Dense packing 
   - Balanced packing
   - WriteOptimized packing
3. **Computation Graph Operations**: Layer mapping, graph splitting, and optimization
4. **ONNX Tools**: Model manipulation, layer extraction, and quantization utilities
5. **Integration Tests**: End-to-end testing with real models (AD, IC, KS, MBV2)

### Running Tests

To run the complete test suite:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_marp.py

# Run with coverage report
pytest --cov=marp --cov-report=html

# Run tests for specific functionality
pytest -k "packing" -v  # Run only packing-related tests
```

### Test Fixtures

The test suite uses parametrized fixtures to ensure comprehensive coverage:

- **Model fixtures**: Tests run against multiple model types (AD, IC, KS, MBV2)
- **Packer fixtures**: All packing strategies are tested systematically
- **Configuration fixtures**: Various hardware and mapping configurations

This approach ensures that each combination of model and packing strategy is thoroughly validated, providing confidence in MARP's reliability across different use cases. algorithms to pack multiple DNN layers into a single matrix written into the AIMC. MARP reduces the number of AIMC cores needed to fully map models OR reduces the number of weight writes onto AIMC memory needed in a full model inference. 

## Reproducing results

To reproduce the reported results, proceed to `marp_results.ipynb`



## Project Structure

```
hwacc_design_garage/
├── README.md                   # Project documentation
├── environment.yml             # Conda environment configuration
├── marp_results.ipynb         # Main results notebook for reproducing experiments
├── naive_mlperftiny.csv       # MLPerf Tiny benchmark results (naive approach)
├── packing_vs_ncores.csv      # Performance comparison data
│
├── marp/                      # Core MARP library
│   ├── compile/               # Compilation and compute modules
│   │   ├── compile.py         # Main compilation logic
│   │   ├── compute.py         # Computation utilities
│   │   └── stimulus_gen.py    # Test stimulus generation
│   │
│   ├── mapping/               # Layer mapping and packing algorithms
│   │   ├── __init__.py
│   │   ├── core.py            # Core mapping functionality
│   │   └── packer_utils.py    # Bin packing utilities
│   │
│   ├── onnx_tools/            # ONNX model processing tools
│   │   ├── onnx_splitter.py   # ONNX model layer splitting
│   │   └── onnx_utils.py      # ONNX utility functions
│   │
│   └── quantization/          # Model quantization tools
│       └── quant.py           # Quantization implementation
│
├── onnx_models/               # Pre-trained and processed ONNX models
│   ├── ad_*.onnx             # Anomaly Detection models
│   ├── ic_*.onnx             # Image Classification models
│   ├── ks_*.onnx             # Keyword Spotting models
│   └── mbv2_*.onnx           # MobileNetV2 variants
│
├── images/                    # Results visualizations and test images
│   ├── *_Balanced.png        # Balanced packing strategy results
│   ├── *_Dense.png           # Dense packing strategy results
│   ├── *_Naive.png           # Naive mapping results
│   ├── *_WriteOptimized.png  # Write-optimized strategy results
│   ├── bin_packing.png       # Bin packing visualization
│   └── test images (*.jpg, *.jpeg)
│
└── tests/                     # Test suite
    ├── __init__.py
    ├── conftest.py           # pytest configuration
    └── test_*.py             # Test modules
```

### Key Components

- **`marp/`**: The core library implementing the MARP algorithm with modules for compilation, mapping, ONNX processing, and quantization
- **`marp_results.ipynb`**: Interactive notebook containing all experiments and results from the paper
- **`onnx_models/`**: Collection of quantized models used in experiments (AD: Anomaly Detection, IC: Image Classification, KS: Keyword Spotting, MBV2: MobileNetV2)
- **`images/`**: Visualization results showing different packing strategies and performance comparisons, but also image inputs for the onnx models.
- **`tests/`**: Comprehensive test suite for validating MARP functionality

