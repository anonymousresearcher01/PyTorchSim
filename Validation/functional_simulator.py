import os
import shlex
import subprocess

import torch
import numpy as np

from extension_backends.llvm_common import LLVMKernelArgs

class FunctionalSimulator():
    def __init__(self, path, key, arg_attributes, args):
        self.path = path
        self.key = key
        self.arg_attributes = arg_attributes
        self.arg_names = arg_attributes.keys()
        self.arg_types = arg_attributes.values()
        self.args = args

    def load_tensor(self, arg, arg_name, arg_attribute, n_call):
        path = os.path.join(self.path, arg_name, f'{n_call}.raw')

        with open(path, 'rb') as f:
            np_array = np.fromfile(f)
            arg = torch.from_numpy(np_array)

    def add_extention(self, name, extension):
        return name + "." + extension

    def run_spike(self, n_call):
        main_path = os.path.join(self.path, self.add_extention(self.key, 'c'))
        main_obj_path = os.path.join(self.path, self.add_extention(self.key + '_main', 'o'))
        kernel_path = os.path.join(self.path, self.add_extention(self.key + '_opt', 'll'))
        kernel_obj_path = os.path.join(self.path, self.add_extention(self.key+'_kernel', 'o'))

        main_compile = f'riscv64-unknown-elf-gcc -march=rv64gcv -c {main_path} -o {main_obj_path}'
        kernel_compile = f'clang -c --target="riscv64" -march=rv64gcv -O2 -nostdlib {kernel_path} -o {kernel_obj_path}'

        target = os.path.join(self.path, self.add_extention(self.key, 'out'))
        link = f'riscv64-unknown-elf-gcc -march=rv64gcv {main_obj_path} {kernel_obj_path} -o {target}'
        run = f'spike --isa rv64gcv /workspace/riscv-pk/build/pk {target} {n_call}'

        main_compile_cmd = shlex.split(main_compile)
        kernel_compile_cmd = shlex.split(kernel_compile)
        link_cmd = shlex.split(link)
        run_cmd = shlex.split(run)

        try:
            subprocess.check_call(main_compile_cmd)
            subprocess.check_call(kernel_compile_cmd)
            subprocess.check_call(link_cmd)
        except subprocess.CalledProcessError as e:
            print("Compile error")
            assert(0)

        try:
            subprocess.check_call(run_cmd)
        except subprocess.CalledProcessError as e:
            print("Spike error")
            assert(0)

        for (arg_name, arg_attribute), arg in zip(self.arg_attributes.items(), self.args):
            if LLVMKernelArgs.is_llvm_arg_out(arg_attribute):
                self.load_tensor(arg, arg_name, arg_attribute, n_call)