import os
import torch
import torch.fx as fx
import oneflow as flow
from torch.fx.node import map_aggregate
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .transform import replace_obj, replace_func, get_attr, torch2oflow



def fx_node_tranform(gm):
    of_gm = to_of_transform(gm)

    enable_graph = os.getenv("ONEDIFF_INFER_COMPILER_USE_GRAPH", "True").lower() in (
        "true",
        "1",
        "t",
    )

    if not enable_graph:
        oneflow_fn = of_gm.forward
    else:
        os.environ["ONEFLOW_MLIR_CSE"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_OPS_WITH_BACKWARD_IMPL"] = "1"
        os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
        os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"
        os.environ["ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
        os.environ["ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
        os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
        os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
        os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"

        class OfGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.fx_md = of_gm
                self.config.enable_cudnn_conv_heuristic_search_algo(False)
                self.config.allow_fuse_add_to_output(True)

            def build(self, *args, **kwargs):
                return self.fx_md(*args, **kwargs)

        of_g = OfGraph()
        oneflow_fn = lambda *args, **kwargs: of_g(*args, **kwargs)

    return oneflow_fn


def to_of_transform(
    gm: torch.fx.GraphModule, tracer_class: type = fx.Tracer
) -> torch.fx.GraphModule:
    name2node = {}
    name2obj = {}
    torch2flow = {}
    of_g = flow.fx.Graph()
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            of_node = of_g.create_node("placeholder", node.target)
            name2node[node.name] = of_node
        elif node.op == "output":
            of_node = of_g.output(node_replace_args(node.args, name2node)[0])
            name2node[node.name] = of_node
        elif node.op == "call_function":
            of_node = of_g.create_node(
                "call_function",
                replace_func(node.target),
                args=node_replace_args(node.args, name2node),
                kwargs=node_replace_args(node.kwargs, name2node),
            )
            name2node[node.name] = of_node
        elif node.op == "call_method":
            of_node = of_g.create_node(
                "call_method",
                node.target,
                args=node_replace_args(node.args, name2node),
                kwargs=node_replace_args(node.kwargs, name2node),
            )
            name2node[node.name] = of_node
        elif node.op == "call_module":
            torch_md = modules[node.target]
            name2obj[node.target] = torch2oflow(torch_md)

            of_node = of_g.create_node(
                "call_module",
                node.target,
                args=node_replace_args(node.args, name2node),
                kwargs=node_replace_args(node.kwargs, name2node),
            )
            name2node[node.name] = of_node
        elif node.op == "get_attr":
            of_node = of_g.create_node("get_attr", node.target)
            name2node[node.name] = of_node
            name2obj[node.target] = get_attr(gm, node, torch2flow)
        else:
            raise ValueError(f"not valid node type{node.foramt_node()}")

    of_gm = flow.fx.GraphModule(name2obj, of_g)
    of_gm.training = False
    of_gm.graph.lint()
    of_gm.recompile()
    return of_gm


def replace_node(node, name2node):
    if isinstance(node, torch.fx.Node):
        return name2node[node.name]
    else:
        return replace_obj(node)


def node_replace_args(args, name2node):
    return map_aggregate(args, lambda node: replace_node(node, name2node))
