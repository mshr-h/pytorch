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
from typing import Optional, Callable

import torch

from .common import device_from_inputs, fake_tensor_unsupported
from .registry import register_backend


log = logging.getLogger(__name__)


@register_backend
@fake_tensor_unsupported
def tvm_relax(
    graph_module,
    example_inputs,
    *,
    options: Optional[MappingProxyType] = MappingProxyType(
        {
            "pipeline_name": None,
            "pipeline_kwargs": {},
        }
    ),
) -> Callable[..., torch.Tensor]:
    """
        from tvm.relax.frontend.torch.dynamo import relax_dynamo

        relax_backend = relax_dynamo()

        return relax_backend(
            graph_module,
            example_inputs,
        )
    """
    # TODOs:
    # 1. support more pipelines
    # 1. support more targets
    # 2. support more options
    # 3. support more devices
    # 1. out-of-the-box performance improvement
    import tvm
    from tvm.relax import build as relax_build
    from tvm.relax.frontend.torch.fx_translator import from_fx

    def to_torch_tensor(nd_tensor):
        """A helper function to transfer a NDArray to torch.tensor."""
        if isinstance(nd_tensor, tvm.nd.NDArray):
            return torch.from_numpy(nd_tensor.numpy())
        elif isinstance(nd_tensor, tvm.ir.Array):
            return tuple(to_torch_tensor(x) for x in nd_tensor)
        else:
            raise ValueError(f"Unsupported type {type(nd_tensor)}")

    def to_tvm_tensor(torch_tensor):
        """A helper function to transfer a torch.tensor to NDArray."""
        if not isinstance(torch_tensor, torch._subclasses.fake_tensor.FakeTensor):
            return tvm.nd.array(torch_tensor.numpy())
        # Fake Tensor
        real_tensor = torch.randn(torch_tensor.shape, dtype=torch_tensor.dtype)
        return tvm.nd.array(real_tensor.numpy())

    graph_module.graph.eliminate_dead_code()

    device = device_from_inputs(example_inputs)

    assert len(example_inputs)

    fake_inputs = []
    if isinstance(example_inputs[0], torch._subclasses.fake_tensor.FakeTensor):
        # Fake tensors
        fake_inputs = example_inputs
    else:
        # Real tensors
        for node in graph_module.graph.nodes:
            if node.op != "placeholder":
                continue
            if "grapharg" not in node.meta:
                continue
            fake_tensor = node.meta["grapharg"].fake_tensor
            if fake_tensor is None:
                continue
            fake_inputs.append(fake_tensor)

    input_info = []
    shape_vars = {}
    for tensor in fake_inputs:
        shape = []
        for s in tensor.shape:
            if isinstance(s, torch.SymInt):
                if str(s) not in shape_vars:
                    shape_vars[str(s)] = tvm.tir.Var(str(s), "int64")
                shape.append(shape_vars[str(s)])
            else:
                shape.append(s)
        input_info.append((shape, tensor.dtype))

    mod = from_fx(graph_module, input_info)

    if device.type == "cuda":
        dev = tvm.cuda(device.index)
        target = tvm.target.cuda()
    else:
        dev = tvm.cpu(0)
        target = tvm.target.Target(llvm_target())

    print("Target: ", target)
    print("Device: ", dev)

    # invoke optimization pipeline.
    pipeline_name = options.get("pipeline_name", None)
    pipeline_kwargs = options.get("pipeline_kwargs", {})
    pipeline_kwargs.update(target=target)
    print("pipeline_kwargs: ", pipeline_kwargs)
    if pipeline_name is None:
        # get default pipeline
        seq = tvm.relax.get_pipeline()
    elif isinstance(pipeline_name, str):
        # lookup by name
        seq = tvm.relax.get_pipeline(pipeline_name, **pipeline_kwargs)
    else:
        seq = pipeline_name

    print("Pipeline: ", pipeline_name)
    print("seq: ", seq)

    mod = mod.with_attr("target", target)
    mod = seq(mod)

    ex = relax_build(mod, target=target)

    vm = tvm.relax.VirtualMachine(ex.mod, device=dev)

    def exec_tvm(*i_args):
        args = [a.contiguous() for a in i_args if isinstance(a, torch.Tensor)]
        vm_args = list()
        for arg in args:
            if arg.dim() != 0:
                if arg.requires_grad:
                    arg = arg.detach()
                vm_args.append(to_tvm_tensor(arg))
        outputs = vm["main"](*vm_args)
        return to_torch_tensor(outputs)

    return exec_tvm


@functools.lru_cache(None)
def llvm_target():
    logical_core_count = os.cpu_count()
    "-num-cores 16"
    if sys.platform == "linux":
        cpuinfo = open("/proc/cpuinfo").read()
        if "avx512" in cpuinfo:
            return f"llvm -mcpu=skylake-avx512 -num-cores {logical_core_count}"
        elif "avx2" in cpuinfo:
            return "llvm -mcpu=core-avx2 -num-cores {logical_core_count}"
    return "llvm -num-cores {logical_core_count}"
