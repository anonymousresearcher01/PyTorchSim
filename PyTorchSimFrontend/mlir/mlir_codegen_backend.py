import dataclasses
import contextlib
import sympy
import itertools
import re
from functools import reduce
from operator import mul
from typing import List
from typing import Dict
import torch
from torch._inductor import dependencies
from torch._inductor.codegen import cpp, wrapper, common
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V, _ops as ops
from torch._inductor.utils import IndentedBuffer
import extension_codecache

from . import mlir_common

class ExtensionWrapperCodegen(wrapper.WrapperCodeGen):
    def __init__(self):
        super().__init__()

    def write_header(self):
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile
                from torch._inductor.codegen.memory_planning import _align as align

                from torch import device, empty, empty_strided
                from {extension_codecache.__name__} import CustomAsyncCompile
                from torch._inductor.select_algorithm import extern_kernels

                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                reinterpret_tensor = torch.ops.aten._reinterpret_tensor
                async_compile = CustomAsyncCompile()

            """
        )

class ExtensionOverrides(common.OpOverrides):
    @staticmethod
    def add(operand1, operand2, tile_size=16):
        return f'arith.addf %{operand1}, %{operand2} : vector<{tile_size}xf32>'

    @staticmethod
    def sub(operand1, operand2, tile_size=16):
        return f'arith.subf %{operand1}, %{operand2} : vector<{tile_size}xf32>'

    @staticmethod
    def mul(operand1, operand2, tile_size=16):
        return f'arith.mulf %{operand1}, %{operand2} : vector<{tile_size}xf32>'

    @staticmethod
    def div(operand1, operand2, tile_size=16):
        return f'arith.divf %{operand1}, %{operand2} : vector<{tile_size}xf32>'

    @staticmethod
    def truediv(operand1, operand2, tile_size=16):
        return f'arith.divf %{operand1}, %{operand2} : vector<{tile_size}xf32>'

    @staticmethod
    def constant(value, dtype, tile_size=16):
        return f'arith.constant {value} : {mlir_common.DTYPE_TO_MLIR[dtype]}'

    @staticmethod
    def exp(operand, tile_size=16):
        return f'math.exp %{operand} : vector<{tile_size}xf32>'

SYMPY_TO_MLIR = {
    sympy.core.mul.Mul: "arith.mulf",
    sympy.core.add.Add: "arith.addf",
}

class MLIRKernel(mlir_common.BaseMLIRKernel):
    overrides = ExtensionOverrides
    newvar_prefix = "%"

    def __init__(self, args=None):
        super().__init__(mlir_common.MLIRKernelArgs())
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.reduction_vars = {}
        self.reduction_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
        self.index_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_idx")
        self.loop_info = {}
        self.load_desc = {}
        self.store_desc = {}

    def get_constant_vector(self, expr):
        constant_vector = [int(expr.coeff(var)) for var in self.itervars]
        return constant_vector

    def add_desc(self, is_load, base_addr, element_size, stride_list, tile_size):
        if is_load:
            key = f"load{len(self.load_desc)}"
            self.load_desc[key] = {
                "base_addr": base_addr,
                "element_size": element_size,
                "stride_list": stride_list,
                "tile_size": tile_size,
                "tile_stride": stride_list[-2:]
            }
        else:
            key = f"store{len(self.store_desc)}"
            self.store_desc[key] = {
                "base_addr": base_addr,
                "element_size": element_size,
                "stride_list": stride_list,
                "tile_size": tile_size,
                "tile_stride": stride_list[-2:]
            }

    def depth_first_traverse(self, expr, buffer, cse):
        child_var = []
        for arg in expr.args:
            child_var.append(self.depth_first_traverse(arg, buffer, cse))

        while len(child_var) >= 3:
            first = child_var.pop(0)
            second = child_var.pop(0)
            first_prefix = "" if first.is_number else "%"
            second_prefix = "" if second.is_number else "%"

            line = f"{SYMPY_TO_MLIR[expr.func]} nsw i64 {first_prefix}{first}, {second_prefix}{second}"
            var = cse.generate(buffer, line)
            var = sympy.symbols(f"{var}")
            child_var.append(var)

        if len(expr.args) == 0:
            return expr

        elif len(child_var) == 2:
            first = child_var[1]
            second = child_var[0]
            first_prefix = "" if first.is_number else "%"
            second_prefix = "" if second.is_number else "%"
            line = f"{SYMPY_TO_MLIR[expr.func]} nsw i64 {first_prefix}{first}, {second_prefix}{second}"
            var = cse.generate(buffer, line)
            var = sympy.symbols(f"{var}")
            return var
        else:
            raise Exception()

    def codegen_nodes(self, nodes, kernel_name):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        _, _, _, self.buffer_types = self.args.mlir_argdefs()
        with self as kernel:
            for node in nodes:
                vars, reduction_vars = kernel.set_ranges(group, reduction_group)
                node.run(vars, reduction_vars)

        src_code = self.codegen_kernel(kernel_name=kernel_name)
        self.meta_kernel()
        return src_code

    def load(self, name: str, index: sympy.Expr):
        index = self.rename_indexing(index)
        index = self.depth_first_traverse(index, self.loads, self.index_cse)
        var = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        line = f"affine.vector_load %{var}[%{index}] : memref<{self.buffer_types[name][1]}x{type_name}>, vector<{self.tile_size}x{type_name}>"
        return self.cse.generate(self.loads, line)

    def store(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        index = self.rename_indexing(index)
        index = self.depth_first_traverse(index, self.stores, self.index_cse)
        var = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        line = f"affine.vector_store %{value}, %{var}[%{index}] : memref<{self.buffer_types[name][1]}x{type_name}>, vector<{self.tile_size}x{type_name}>"
        self.cse.generate(self.stores, line, assignment = False)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError()

    def store_reduction(self, name, index, value):
        raise NotImplementedError()

    def codegen_loops(self):
        code = common.BracesBuffer()
        # Loop body part
        loops = [LoopLevel(var, size, idx, tile_size=self.tile_size) for idx, (var, size) in enumerate(zip(self.itervars, self.ranges))]
        loops, reductions = [LoopNest(loops[: self.reduction_depth]),
                             LoopNest(loops[self.reduction_depth :])]
        reductions.mark_reduction(self.reduction_vars)

        with contextlib.ExitStack() as stack:
            if self.reduction_vars:
                raise NotImplementedError()
            for loop in loops.loops:
                loop_lines = loop.lines()
                if loop_lines is None:
                    return
                code.writelines(loop_lines)
                stack.enter_context(code.indent())
                with contextlib.ExitStack() as stack_outer:
                    code.splice(self.reduction_prefix)
                    with contextlib.ExitStack() as stack:
                        code.splice(self.loads)
                        code.splice(self.compute)
                        code.splice(self.stores)
                    code.splice(self.reductions_suffix)
        code.writeline(f"return")
        return code

    def codegen_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        arg_defs, _, _, _ = self.args.mlir_argdefs()
        code = self._codegen_kernel(arg_defs, kernel_name)
        return code.getvalue()

    def meta_kernel(self):
        wrapper = V.graph.wrapper_code
        _, _, arg_attributes, _ = self.args.mlir_argdefs()
        wrapper.add_import_once('\nprint(f\'Wrapper Codegen Path = {__file__}\')')
        wrapper.add_import_once(f'\nfrom extension_codecache import CustomAsyncCompile')
        wrapper.add_import_once(f'\ncustom_async_compile = CustomAsyncCompile()')
        # Dump loop and load/store information
        wrapper.add_import_once(f"loop_info = {self.loop_info}")
        wrapper.add_import_once(f"load_tile_info = {self.load_desc}")
        wrapper.add_import_once(f"store_tile_info = {self.store_desc}")
        wrapper.add_import_once(f"arg_attributes = {arg_attributes}")


    def call_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        _, call_args, _, _ = self.args.mlir_argdefs()
       # generate the code to call this
        wrapper.generate_kernel_call(kernel_name, call_args, cuda=False)

    def _codegen_kernel(self, arg_defs, kernel_name):
        arg_defs = ",\n".ljust(25).join(arg_defs)
        code = common.BracesBuffer()

        # Todo. kernel name custom
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"
        code.writeline(f'func.func @{kernel_decl_name}({arg_defs})')
        with code.indent():
            for old, new in self.args.aliases():
                code.writeline(f"auto {old} = {new};")
            # Loop body part
            code.splice(self.codegen_loops())
        return code


    def set_ranges(self, lengths, reduction_lengths):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(
                reduction_lengths
            ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [sympy.Symbol(f"index{n}") for n in range(len(self.ranges))]
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr
    size: sympy.Expr
    idx: int
    start: int = 0
    tile_size: int = 4
    reduction_vars: Dict[str, str] = None

    def lines(self):
        loop_index = self.idx
        line = f"affine.for %index{loop_index} = {self.start} to {self.size} step {self.tile_size}"

        return [line]

@dataclasses.dataclass
class LoopNest:
    loops: List[LoopLevel]

    def __bool__(self):
        return bool(self.loops)

    def mark_reduction(self, reduction_vars):
        for loop in self.loops:
            loop.reduction_vars = reduction_vars

    def mark_parallel(self, par_depth):
        loops = self.loops
        loops[0].parallel = par_depth
        for i in range(1, par_depth):
            loops[i].collapsed = True
        loops[0].simd = loops[par_depth - 1].simd

class MLIRScheduling(BaseScheduling):
    count = 0
    target_kernel = MLIRKernel
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._scheduling = cpp.CppScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        return False

    def can_fuse_horizontal(self, node1, node2):
        return False

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_nodes(self, nodes):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group
        ex_kernel = self.target_kernel()

        kernel_name = f"extension_kernel_{self.count}"
        self.count += 1
        src_code = ex_kernel.codegen_nodes(nodes, kernel_name)
        self.define_kernel(src_code, kernel_name)
        ex_kernel.call_kernel(kernel_name)

    def codegen_sync(self):
        pass

    def flush(self):
        self._scheduling.flush()

    def define_function(self, kernel):
        code = kernel.def_function()
        if code is not None:
            wrapper = V.graph.wrapper_code
            wrapper.header.writeline(code)

    def define_kernel(self, src_code, kernel_name):
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            wrapper.src_to_kernel[src_code] = kernel_name

            codecache_def = IndentedBuffer()
            codecache_def.writeline("custom_async_compile.mlir('''")
            codecache_def.splice(src_code)
            codecache_def.writeline("''', ")
            codecache_def.writeline("loop_info=loop_info,")
            codecache_def.writeline("load_tile_info=load_tile_info,")
            codecache_def.writeline("store_tile_info=store_tile_info,")
            codecache_def.writeline("arg_attributes=arg_attributes)")

            wrapper.define_kernel(kernel_name, codecache_def.getvalue(), cuda=False)
        return kernel_name