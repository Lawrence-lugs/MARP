"""Hardware constants and enumerations for the QRAcc accelerator."""

from __future__ import annotations

from enum import IntEnum

# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

DEFAULT_CORE_SIZE: tuple[int, int] = (256, 256)
"""Default AIMC core dimensions (rows, columns)."""

DWC_CORE_SIZE: int = 32
"""Weight-storage width for depthwise convolution cores."""

NUM_BANK_COLS: int = 32
"""Number of columns per weight-memory bank."""

# ---------------------------------------------------------------------------
# CSR address space
# ---------------------------------------------------------------------------

CSR_BASE_ADDR: int = 0x00000010
"""Base address of the main configuration CSR block."""

CSR_BASE_ADDR_HEX: str = "00000010"
"""Hex-string form of *CSR_BASE_ADDR* (for assembly emission)."""

DATA_WRITE_ADDR: str = "00000100"
"""Hex-string address used when writing data arrays to memory."""

# CSR register offsets (from base address)
CSR_REG_CONFIG: int = 1
CSR_REG_IFMAP_DIMS: int = 2
CSR_REG_OFMAP_DIMS: int = 3
CSR_REG_CHANNELS: int = 4
CSR_REG_OFFSETS: int = 5
CSR_REG_PADDING: int = 6


# ---------------------------------------------------------------------------
# Hardware trigger commands
# ---------------------------------------------------------------------------

class Trigger(IntEnum):
    """Trigger values written to the main CSR to start operations."""

    IDLE = 0
    LOAD_ACTIVATION = 1
    LOADWEIGHTS = 2
    COMPUTE_ANALOG = 3
    COMPUTE_DIGITAL = 4
    READ_ACTIVATION = 5
    LOADWEIGHTS_DIGITAL = 6
    LOAD_SCALER = 7


# Legacy string→int mapping (kept for backward compat in `make_trigger_write`)
TRIGGER_ENUM_MAP: dict[str, int] = {
    f"TRIGGER_{t.name}": t.value for t in Trigger
}

# ---------------------------------------------------------------------------
# Config-word bit-field positions and masks
# ---------------------------------------------------------------------------

MASK_1: int = 0x1
MASK_4: int = 0xF
MASK_8: int = 0xFF
MASK_16: int = 0xFFFF
