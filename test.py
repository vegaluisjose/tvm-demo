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

from os import path
import numpy as np
import subprocess as sp

import tflite
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
from tvm.contrib.download import download_testdata


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


def compile_verilog(files):
    name = "Top"
    verilator = util.which("verilator")
    share_dir = path.realpath(path.join(path.dirname(verilator), "..", "share"))
    inc_dir = path.join(share_dir, "verilator", "include")
    target_dir = util.tempdir().temp_dir
    wno = ["BLKANDNBLK", "PINMISSING", "STMTDLY", "WIDTH", "UNOPTFLAT"]
    wno = ["-Wno-{}".format(w) for w in wno]
    cmd = []
    cmd.append(verilator)
    cmd.append("--cc")
    cmd.append("--prefix")
    cmd.append(name)
    cmd.append("--Mdir")
    cmd.append(target_dir)
    cmd = cmd + wno + files
    try:
        sp.run(cmd, check=True)
        cfiles = [
            "{}__Slow.cpp".format(name),
            "{}__Syms.cpp".format(name),
            "{}.cpp".format(name),
        ]
        cfiles = [path.join(target_dir, f) for f in cfiles]
        cfiles.append(path.join(inc_dir, "verilated.cpp"))
        return ["-I" + target_dir, "-I" + inc_dir] + cfiles
    except:
        print("Verilator error")
        raise


def update_lib(lib, backend):
    test_dir = path.dirname(path.realpath(path.expanduser(__file__)))
    source_dir = path.join(test_dir, "..", "..", "..")
    contrib_path = path.join(source_dir, "src", "runtime", "contrib")

    files = ["adder.v", "driver.v"]
    files = [path.join(test_dir, "hardware", "adder", f) for f in files]

    verilog_opts = compile_verilog(files)
    verilog_opts += [
        "-I" + test_dir,
        "-I" + path.join(test_dir, "driver"),
        path.join(test_dir, "hardware", "accel.cc"),
        path.join(test_dir, "driver", "verilator_driver.cc"),
    ]
    cc_opts = ["-I" + test_dir, path.join(test_dir, "reference.cc")]

    opts = verilog_opts if backend == "verilator" else cc_opts

    kwargs = {}
    kwargs["options"] = ["-O2", "-std=c++14", "-I" + contrib_path] + opts
    tmp_path = util.tempdir()
    lib_name = "lib.so"
    lib_path = tmp_path.relpath(lib_name)
    lib.export_library(lib_path, fcompile=False, **kwargs)
    lib = runtime.load_module(lib_path)

    return lib


def extract(file):
    import tarfile

    if file.endswith("tgz") or file.endswith("gz"):
        dir_path = path.dirname(file)
        tar = tarfile.open(file)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError("Could not decompress the file: " + file)


def build_bias_add_program(xshape, bshape, dtype):
    x = relay.var("x", shape=xshape, dtype=dtype)
    bias = relay.var("bias", shape=bshape, dtype=dtype)
    z = relay.nn.bias_add(x, bias, axis=3)
    f = relay.Function([x, bias], z)
    mod = tvm.IRModule()
    mod["main"] = f
    return mod


def get_mobilenet(shape, dtype):
    input_tensor = "input"
    model_url = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz"
    model_path = download_testdata(
        model_url, "mobilenet_v1_1.0_224_quant.tgz", module=["tf", "official"]
    )
    model_dir = path.dirname(model_path)
    extract(model_path)
    tflite_model_file = path.join(
        model_dir, "mobilenet_v1_1.0_224_quant.tflite"
    )
    tflite_model_buf = open(tflite_model_file, "rb").read()
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    return relay.frontend.from_tflite(
        tflite_model,
        shape_dict={input_tensor: shape},
        dtype_dict={input_tensor: dtype},
    )


def partition(mod, compiler, op):
    mod = WhiteListAnnotator([op], compiler)(mod)
    mod = transform.PartitionGraph()(mod)
    return mod


def compile_prog(mod, params=None, backend=None):
    with relay.build_config(opt_level=3):
        exe = relay.vm.compile(mod, target="llvm", params=params)
        code, lib = exe.save()
        lib = update_lib(lib, backend)
        return runtime.vm.Executable.load_exec(code, lib)


def run(exe, inputs):
    ctx = tvm.cpu()
    vm = runtime.vm.VirtualMachine(exe, ctx)
    return vm.run(**inputs)


def run_bias_add(exe, xshape, bshape, dtype):
    x_data = np.random.randint(5, size=xshape, dtype=dtype)
    bias_data = np.random.randint(5, size=bshape, dtype=dtype)
    ref = x_data + bias_data.reshape((1, 1, bshape[0]))
    inputs = {"x": x_data, "bias": bias_data}
    out = run(exe, inputs)
    tvm.testing.assert_allclose(out.asnumpy(), ref, rtol=1e-5, atol=1e-5)


def run_mobilenet(exe0, exe1, shape, dtype):
    i_data = np.random.randint(5, size=shape, dtype=dtype)
    inputs = {"input": i_data}
    out0 = run(exe0, inputs)
    out1 = run(exe1, inputs)
    tvm.testing.assert_allclose(
        out0.asnumpy(), out1.asnumpy(), rtol=1e-5, atol=1e-5
    )


def test_bias_add(compiler, backend):
    dtype = "int32"
    xshape = (1, 112, 112, 32)
    bshape = (32,)
    mod = build_bias_add_program(xshape, bshape, dtype)
    mod = partition(mod, compiler, "nn.bias_add")
    exe = compile_prog(mod, params=None, backend=backend)
    run_bias_add(exe, xshape, bshape, dtype)


def test_mobilenet(compiler, backend):
    dtype = "int8"
    shape = (1, 224, 224, 3)
    mod0, param = get_mobilenet(shape, dtype)
    mod1 = partition(mod0, compiler, "nn.bias_add")
    exe0 = compile_prog(mod0, param, backend=None)
    exe1 = compile_prog(mod1, param, backend=backend)
    run_mobilenet(exe0, exe1, shape, dtype)


if __name__ == "__main__":
    compiler = "ccompiler"
    test_bias_add(compiler, "verilator")
    test_mobilenet(compiler, "verilator")
