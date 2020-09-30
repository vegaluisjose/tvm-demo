# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Unit tests for graph partitioning."""
# pylint: disable=not-callable
import os
import sys

import numpy as np

import tvm
import tvm.relay.testing
import tvm.relay.op as reg
from tvm import relay
from tvm import runtime
from tvm.relay import transform
from tvm.contrib import util
from tvm.relay.backend import compile_engine
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.op.contrib.register import get_pattern_table
from tvm.relay.build_module import bind_params_by_name


# Leverage the pass manager to write a simple white list based annotator
@transform.function_pass(opt_level=0)
class WhiteListAnnotator:
    def __init__(self, op_list, compiler):
        assert isinstance(op_list, (list, tuple, set))
        self.op_list = op_list
        self.compiler = compiler

    def transform_function(self, func, mod, ctx):

        annotator = self

        class Annotator(tvm.relay.ExprMutator):
            def visit_call(self, call):
                op_name = call.op.name
                if op_name in annotator.op_list:
                    new_args = []
                    for arg in call.args:
                        ann = compiler_begin(
                            super().visit(arg), annotator.compiler
                        )
                        new_args.append(ann)
                    new_call = relay.Call(
                        call.op, new_args, call.attrs, call.type_args
                    )
                    return compiler_end(new_call, annotator.compiler)
                else:
                    return super().visit_call(call)

        return Annotator().visit(func)


def update_lib(lib):
    test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.join(test_dir, "..", "..", "..")
    contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

    kwargs = {}
    kwargs["options"] = ["-O2", "-std=c++14", "-I" + contrib_path]
    tmp_path = util.tempdir()
    lib_name = "lib.so"
    lib_path = tmp_path.relpath(lib_name)
    lib.export_library(lib_path, fcompile=False, **kwargs)
    lib = runtime.load_module(lib_path)

    return lib


def prog_bias_add(xshape, bshape, dtype):
    x = relay.var("x", shape=xshape, dtype=dtype)
    bias = relay.var("bias", shape=bshape, dtype=dtype)
    z = relay.nn.bias_add(x, bias, axis=3)
    f = relay.Function([x, bias], z)
    mod = tvm.IRModule()
    mod["main"] = f
    return mod


def prog_add(shape, dtype):
    x = relay.var("x", shape=shape, dtype=dtype)
    y = relay.var("y", shape=shape, dtype=dtype)
    z = x + y
    f = relay.Function([x, y], z)
    mod = tvm.IRModule()
    mod["main"] = f
    return mod


def partition(mod, compiler):
    mod = WhiteListAnnotator(["nn.bias_add"], compiler)(mod)
    mod = transform.PartitionGraph()(mod)
    return mod


def compile_prog(mod):
    with relay.build_config(opt_level=3):
        exe = relay.vm.compile(mod, target="llvm", params=None)
        code, lib = exe.save()
        lib = update_lib(lib)
        return runtime.vm.Executable.load_exec(code, lib)


def run(exe, inputs, ref):
    ctx = tvm.cpu()
    vm = runtime.vm.VirtualMachine(exe, ctx)
    out = vm.run(**inputs)
    tvm.testing.assert_allclose(out.asnumpy(), ref, rtol=1e-5, atol=1e-5)


def run_add(exe, shape, dtype):
    x_data = np.random.randint(5, size=shape, dtype=dtype)
    y_data = np.random.randint(5, size=shape, dtype=dtype)
    ref = x_data + y_data
    inputs = {"x": x_data, "y": y_data}
    run(exe, inputs, ref)

def run_bias_add(exe, xshape, bshape, dtype):
    x_data = np.random.randint(5, size=xshape, dtype=dtype)
    bias_data = np.random.randint(5, size=bshape, dtype=dtype)
    ref = x_data + bias_data.reshape((1, 1, bshape[0]))
    inputs = {"x": x_data, "bias": bias_data}
    run(exe, inputs, ref)


def test_add(compiler):
    dtype = "int32"
    shape = (8, 8)
    mod = prog_add(shape, dtype)
    mod = partition(mod, compiler)
    exe = compile_prog(mod)
    run_add(exe, shape, dtype)


def test_bias_add(compiler):
    dtype = "int32"
    xshape = (1, 112, 112, 32)
    bshape = (32,)
    mod = prog_bias_add(xshape, bshape, dtype)
    mod = partition(mod, compiler)
    exe = compile_prog(mod)
    run_bias_add(exe, xshape, bshape, dtype)


if __name__ == "__main__":
    compiler = "ccompiler"
    test_add(compiler)
    test_bias_add(compiler)