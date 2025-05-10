# mypy: ignore-errors

"""
"""

import functools
import importlib
import logging
import os
import sys
import tempfile
from types import MappingProxyType
from typing import Optional

import torch

from .common import device_from_inputs, fake_tensor_unsupported
from .registry import register_backend


log = logging.getLogger(__name__)


@register_backend
@fake_tensor_unsupported
def tvm_relax(
    gm,
    example_inputs,
    *,
    options: Optional[MappingProxyType] = MappingProxyType(
        {"opt_level": 3}
    ),
):
    print("Using TVM Relax backend")
    for idx, i in enumerate(example_inputs):
        print(f"{idx}: {i.shape}")
    print(gm.graph)
    return gm.forward
