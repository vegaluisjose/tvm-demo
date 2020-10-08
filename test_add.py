# this is for regression purposes, and test that default
# add is not breaking
import os

import numpy as np

import tvm
from tvm import relay
from tvm.contrib import util
from tvm import runtime
from test import partition, compile_prog, run


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


def build_add_program(shape, dtype):
    x = relay.var("x", shape=shape, dtype=dtype)
    y = relay.var("y", shape=shape, dtype=dtype)
    z = x + y
    f = relay.Function([x, y], z)
    mod = tvm.IRModule()
    mod["main"] = f
    return mod


def run_add(exe, shape, dtype):
    x_data = np.random.randint(5, size=shape, dtype=dtype)
    y_data = np.random.randint(5, size=shape, dtype=dtype)
    ref = x_data + y_data
    inputs = {"x": x_data, "y": y_data}
    out = run(exe, inputs)
    tvm.testing.assert_allclose(out.asnumpy(), ref, rtol=1e-5, atol=1e-5)


def test_add(compiler):
    dtype = "int32"
    shape = (8, 8)
    mod = build_add_program(shape, dtype)
    mod = partition(mod, compiler, "add")
    exe = compile_prog(mod)
    run_add(exe, shape, dtype)


if __name__ == "__main__":
    compiler = "ccompiler"
    test_add(compiler)
