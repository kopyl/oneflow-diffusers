import os
import torch
import oneflow as flow

from .utils.patch_for_compiler import *  # TODO:
from .transform.custom_transform import register
from .with_oneflow_compile import oneflow_compile, oneflow_load_compiled

from .with_fx_interpreter import OneFlowInterpreter
from .with_fx_graph import fx_node_tranform


def oneflow_backend(gm, example_inputs):
    with_interp = os.getenv(
        "ONEDIFF_INFER_COMPILER_USE_INTERPRETER", "False"
    ).lower() in (
        "true",
        "1",
        "t",
    )
    if not with_interp:
        transformed_fn = fx_node_tranform(gm)

    def wrapped_forward(*args, **kwargs):
        args = [flow.utils.tensor.from_torch(a) for a in args]
        if with_interp:
            output = OneFlowInterpreter(gm, garbage_collect_values=False).run(
                *args, **kwargs
            )
        else:
            output = transformed_fn(*args, **kwargs)
        if isinstance(output, tuple):
            return tuple(flow.utils.tensor.to_torch(i) for i in output)
        return flow.utils.tensor.to_torch(output)

    return wrapped_forward
