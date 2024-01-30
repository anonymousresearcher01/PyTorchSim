import torch

from torch._inductor.utils import IndentedBuffer

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

        self.write_header()

    def write_header(self):
        pass

    def type_to_string(self, arg):
        d_type = arg.dtype
        if d_type == torch.float32:
            return 'float'
        elif d_type == torch.int32:
            return 'int'

    def write_exit(self):
        self.writeline('asm volatile(')
        with self.code.indent():
            self.writeline(f'"li a7, 93"')
            self.writeline(f'"li a0, 0"')
            self.writeline(f'"ecall"')
        self.writeline(f'){self.ending}')

    def generate_kernel_declare(self, name, args):
        self.args_type = [f'{self.type_to_string(arg)}' for arg in args]
        args_type_p = [f'{arg_type}*' for arg_type in self.args_type]

        self.writeline(f"void {name}({', '.join(args_type_p)}){self.ending}{self.newline}")

    def generate_args_define(self, args):
        self.shapes = [list(arg.view(-1).shape)[0] for arg in args]
        self.args_name = [f'arg_{idx}' for idx in range(len(args))]

        for idx, arg in enumerate(args):
            self.writeline(f'{self.args_type[idx]} {self.args_name[idx]}[{self.shapes[idx]}]{self.ending}')

    def generate_nostdlib_start(self, kernel_name):
        self.writeline(f'{self.newline}void _start() {self.open_bracket}{self.newline}')

        with self.code.indent():
            self.writeline(f"{kernel_name}({', '.join(self.args_name)}){self.ending}{self.newline}")
            self.write_exit()
        self.writeline(self.closed_bracket)

    def writeline(self, line):
        self.code.writeline(line)

    def generate(self, args):
        kernel_name = "kernel"

        args = [arg.to('cpu') if arg.device != torch.device('cpu') else arg for arg in args]

        self.generate_kernel_declare(kernel_name, args)
        self.generate_args_define(args)
        self.generate_nostdlib_start(kernel_name)

        return self.code.getvalue()