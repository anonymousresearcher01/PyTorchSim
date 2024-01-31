import os
import torch

from torch._inductor.utils import IndentedBuffer

from extension_backends.llvm_common import LLVMKernelArgs

class LLVMKernelCallerCodeGen():
    """
    Generate C that calls the llvm kernel.
    """

    def __init__(self):
        super().__init__()
        self.code = IndentedBuffer()
        self.ending = ";"
        self.open_bracket = "{"
        self.closed_bracket = "}"
        self.newline = "\n"

    def write_header(self):
        self.writeline('#include <stdio.h>')
        self.writeline('#include <stdlib.h>')
        if self.validation:
            self.writeline('#include <string.h>')
            self.writeline('#include <fcntl.h>')

    def type_to_string(self, arg):
        d_type = arg.dtype
        if d_type == torch.float32:
            return 'float'
        elif d_type == torch.int32:
            return 'int'

    def is_in_arg(self, arg_name):
        value = self.arg_attributes[arg_name]
        return (LLVMKernelArgs.LLVM_ARGS_IN & value) | (LLVMKernelArgs.LLVM_ARGS_INOUT & value)

    def is_out_arg(self, arg_name):
        value = self.arg_attributes[arg_name]
        return (LLVMKernelArgs.LLVM_ARGS_OUT & value) | (LLVMKernelArgs.LLVM_ARGS_INOUT & value)

    def preprocess_info(self, kernel_name, arg_attributes, args):
        self.kernel_name = kernel_name
        self.arg_attributes = arg_attributes
        self.args = args
        self.n_arg = len(args)
        self.args_name = list(self.arg_attributes.keys())
        self.args_type = [f'{self.type_to_string(arg)}' for arg in args]
        self.shapes = [list(arg.view(-1).shape)[0] for arg in args]

    def load_arg(self):
        self.writeline(f'char file_name[10]{self.ending}')
        self.writeline(f'char path[100]{self.ending}')
        self.writeline(f'sprintf(file_name, "%d.raw", n_call){self.ending}')
        for i in range(len(self.args)):
            if self.is_in_arg(self.args_name[i]):
                path = os.path.join(self.dump_path, f'arg{i}_1/')
                self.writeline(f'strcpy(path, "{path}"){self.ending}')
                self.writeline(f'strcat(path, file_name){self.ending}')
                self.writeline(f'if(load_arg({self.args_name[i]}, sizeof({self.args_name[i]}), path) == -1){self.open_bracket}')
                with self.code.indent():
                    self.writeline(f'return -1{self.ending}')
                self.writeline(self.closed_bracket)

    def dump_arg(self):
        self.writeline(f'sprintf(file_name, "%d.raw", n_call){self.ending}')
        for i in range(len(self.args)):
            if self.is_out_arg(self.args_name[i]):
                path = os.path.join(self.dump_path, f'{self.args_name[i]}/')
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                self.writeline(f'strcpy(path, "{path}"){self.ending}')
                self.writeline(f'strcat(path, file_name){self.ending}')
                self.writeline(f'if(dump_arg({self.args_name[i]}, sizeof({self.args_name[i]}), path) == -1){self.open_bracket}')
                with self.code.indent():
                    self.writeline(f'return -1{self.ending}')
                self.writeline(self.closed_bracket)

    def write_exit(self):
        self.writeline(f'return 0{self.ending}')

    def generate_kernel_declare(self):
        args_type_p = [f'{arg_type}*' for arg_type in self.args_type]

        self.writeline(f"void {self.kernel_name}({', '.join(args_type_p)}){self.ending}{self.newline}")

    def generate_args_define(self):
        for idx, arg in enumerate(self.args):
            self.writeline(f'{self.args_type[idx]} {self.args_name[idx]}[{self.shapes[idx]}]{self.ending}')

    def generate_load_dump_fn(self):
        self.writeline(f'{self.newline}int load_arg(void *arg, int size, const char *path) {self.open_bracket}')
        with self.code.indent():
            self.writeline(f'int fd = open(path, 0x00000000){self.ending}')
            self.writeline(f'if (fd == -1) {self.open_bracket}')
            with self.code.indent():
                self.writeline(f'return -1{self.ending}')
            self.writeline(self.closed_bracket)

            self.writeline(f'if (read(fd, arg, size) == -1) {self.open_bracket}')
            with self.code.indent():
                self.writeline(f'return -1{self.ending}')
            self.writeline(self.closed_bracket)
            self.writeline(f'close(fd){self.ending}')
            self.writeline(f'return 0{self.ending}')
        self.writeline(self.closed_bracket)

        self.writeline(f'{self.newline}int dump_arg(void *arg, int size, const char *path) {self.open_bracket}')
        with self.code.indent():
            self.writeline(f'int fd = open(path, 0x00000001 | 0x00000040, 0644){self.ending}')
            self.writeline(f'if (fd == -1) {self.open_bracket}')
            with self.code.indent():
                self.writeline(f'return -1{self.ending}')
            self.writeline(self.closed_bracket)

            self.writeline(f'if (write(fd, arg, size) == -1) {self.open_bracket}')
            with self.code.indent():
                self.writeline(f'return -1{self.ending}')
            self.writeline(self.closed_bracket)
            self.writeline(f'close(fd){self.ending}')
            self.writeline(f'return 0{self.ending}')
        self.writeline(self.closed_bracket)

    def generate_main(self):
        self.writeline(f'{self.newline}int main(int argc, char *argv[]) {self.open_bracket}{self.newline}')

        with self.code.indent():
            if self.validation:
                self.writeline(f'int n_call = atoi(argv[1]){self.ending}{self.newline}')
                self.load_arg()
                self.writeline(self.newline)

            self.writeline(f"{self.kernel_name}({', '.join(self.args_name)}){self.ending}{self.newline}")

            if self.validation:
                self.dump_arg()

            self.write_exit()
        self.writeline(self.closed_bracket)

    def writeline(self, line):
        self.code.writeline(line)

    def generate(self, path, arg_attributes, validation, args):
        self.dump_path = path
        self.validation = validation
        kernel_name = "kernel"

        args = [arg.to('cpu') if arg.device != torch.device('cpu') else arg for arg in args]
        self.preprocess_info(kernel_name, arg_attributes, args)

        self.write_header()
        self.generate_kernel_declare()
        self.generate_args_define()
        
        if self.validation:
            self.generate_load_dump_fn()
        self.generate_main()

        return self.code.getvalue()