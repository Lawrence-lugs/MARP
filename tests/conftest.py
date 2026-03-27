"""Shared pytest fixtures for MARP tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "onnx_models"


@pytest.fixture
def model_dir() -> Path:
    """Absolute path to the ``onnx_models/`` directory."""
    return MODEL_DIR