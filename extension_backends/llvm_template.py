import functools
import itertools
from typing import List, Optional, cast
from unittest.mock import patch

import torch
from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.codegen.common import ChoiceCaller
from torch._inductor.codegen.common import Kernel
from torch._inductor.codegen.common import OpOverrides
from torch._inductor.ir import TensorBox
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import TemplateBuffer
from torch._inductor.utils import override_lowering
from torch._inductor.lowering import register_lowering, get_overloads, lowerings
from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller
from torch._inductor.autotune_process import TensorMeta
from torch._inductor.virtualized import V
from torch._inductor.kernel.mm_common import mm_args

from extension_backends.llvm_autotune import LLVMBenchmarkRequest
from extension_backends.llvm_common import LLVMKernelArgs

aten = torch.ops.aten

class LLVMTemplateKernel(Kernel):
    overrides = OpOverrides
    def __init__(self, kernel_name) -> None:
        super().__init__(LLVMKernelArgs())
        self.kernel_name = kernel_name
        self.named_nodes = {}

    def meta_kernel(self):
        wrapper = V.graph.wrapper_code
        _, _, arg_attributes = self.args.llvm_argdefs()
        wrapper.add_import_once('\nprint(f\'Wrapper Codegen Path = {__file__}\')')
        wrapper.add_import_once(f'\nfrom extension_codecache import CustomAsyncCompile')
        wrapper.add_import_once(f'\ncustom_async_compile = CustomAsyncCompile()')
        # Dump loop and load/store information
        wrapper.add_import_once(f"loop_info = dict()")
        wrapper.add_import_once(f"load_tile_info = dict()")
        wrapper.add_import_once(f"store_tile_info = dict()")
        wrapper.add_import_once(f"arg_attributes = {arg_attributes}")

    def call_kernel(self):
        """
        Generates code to call the kernel through V.graph.wrapper_code.
        used from within torch._inductor.wrapper.WrapperCodeGen
        """
        wrapper = V.graph.wrapper_code
        _, call_args, _ = self.args.python_argdefs()
        wrapper.generate_kernel_call(
            self.kernel_name,
            call_args,
            cuda=False,
        )

    def def_kernel(
        self,
        inputs: List[IRNode],
        outputs: List[IRNode],
        names_str: str = "",
        input_reorder: Optional[List[int]] = None,
    ) -> str:
        names = [x.strip() for x in names_str.strip().split(",")]
        if len(inputs) + len(outputs) != len(names):
            raise RuntimeError(
                f"{len(inputs) + len(outputs)=} != {len(names)=}, {inputs=}, {outputs=}, {names=}"
            )

        if input_reorder is not None:
            assert len(inputs) == len(input_reorder)
        else:
            input_reorder = list(range(len(inputs)))

        for idx in input_reorder:
            name = names[idx]
            node = inputs[idx]
            if node is not None:
                self.named_nodes[name] = node
                self.args.input_buffers[node.get_name()] = name

        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                self.named_nodes[name] = node
                self.args.output_buffers[node.get_name()] = name

class LLVMTemplateCaller(CUDATemplateCaller):
    def __str__(self):
        return f"LLVMTemplateCaller(source_file={self.bmreq.source_file})"

    def call_name(self) -> str:
        return f"llvm_template_kernels.{self.name}"

class LLVMTemplate(KernelTemplate):
    index_counter = itertools.count()

    def __init__(self, name, input_nodes, layout, input_reorder = None):
        """
        Baseclass for LLVM Templates, derived from KernelTemplate. Not to be instantiated directly.

        Args:
            name (str): The name of the CUDATemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.

        """
        super().__init__(name)
        self.input_nodes = input_nodes
        self.output_node: Buffer = Buffer("buf_out", layout)
        self.input_reorder = input_reorder
        self.layout = layout

    def generate(self, **kwargs) -> ChoiceCaller:
        kernel_name = f"llvm_{self.name}"
        with patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)):
            kernel  = LLVMTemplateKernel(kernel_name=kernel_name)
            code = self.render(kernel=kernel, **kwargs)

        kernel_hash_name = f"llvm_{self.name}_{next(self.index_counter)}"
        extra_args = []
        # create the BenchmarkRequest
        bmreq = LLVMBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: TemplateBuffer,
            epilogue_nodes: Optional[List[IRNode]] = None,
        ):
            kernel = LLVMTemplateKernel(
                kernel_name="KERNEL_NAME",
            )
            render = functools.partial(
                self.render,
                kernel=kernel,
                template_buffer_node=template_node,
                epilogue_nodes=epilogue_nodes,
                **kwargs,  # includes "op" argument in case of CUTLASSGemmTemplate
            )
            return kernel, render

        return LLVMTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
        )

    def render(self, **kwargs) -> str:
        raise NotImplementedError

class LLVMGemmTemplate(LLVMTemplate):
    def __init__(self, input_nodes, layout, input_reorder=None):
        super().__init__("LLVMGEMM", input_nodes, layout, input_reorder)

    def render(self,
               kernel: LLVMTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               **kwargs):
        code = """
......................................................  .       . ........   .....    .  ............................................................
..............................................................:::::--=*+=++==-:......    .. ..........................................................
.........................................................:-=+++**+++**#%*+*******+-:...  ..  ....................................................  ...
......................................................-=************#**##*##########*+=-..............................................................
...................................................-=+*#*######****+++++*##%%###%######*++-...........................................................
.................................................-+*****######****++==++*###%%%%%#######*++*+:........................................................
...............................................-++++*****###*#**++=====+**##%%%##%##*******++*+-......................................................
.............................................-+++++*++++****#**+++======+*##%########*+++****+*#+-....................................................
...........................................:=+*+****++++*+*++**++===---==+**#####******+++********=:..................................................
..........................................-++**##*###***+##****+==-=---==++*****#*****+*******#*++*+=:................................................
.........................................=+**##%#####*######***+==------==++++**+#*****+*#####*##****+:...............................................
........................................=+**##%%#######%%####**==----:--==++++***+***###**#########***+-..............................................
.......................................=****##%#####%#%%%%%#*+==---------==++*********##%###%%%%##%###*+-.............................................
......................................:***##%%%##%%%%%%%%%#*++=---:::::::-==++*#*****####%%##%%%%#%%###*=:............................................
......................................=**##%%%%%%%%%%%%%%#*++=-----::::::--==+**#*########%%%#%%%%#%%%%#+:............................................
......................................=**#%%@@@@%%%%%%%%%#**+++===----:----==+**#%%#%#%%%#%%%%#%%@%%@%%%*=............................................
......................................=*#%%%@@@%%%@%%%%%**+++++++===------==++**###%####%%%%%%%%%%@%%@%@#+............................................
......................................-*#%%@%@@%@%%%%%#+===--===++++========++++*+****+**##%###%%%%@%%%%#*............................................
......................................:+##%@@%@@@%%%##*=----::--============++++===+====++**#**#%%%%%%@%#+:...........................................
......................................:=*#%@@%%%###*#*+=--==----======--:--==++====++++=++++++++*#*#%%@#*=:...........................................
......................................::-*#%%#%*+++++**#*+##%%*++====--:::-====+***#%##++**++=====-=%%#*-.............................................
......................................:::+##+-+========---====--===----::::-========------==-------:*#*-..............................................
....................................:::::-#*=:--------=====+++==-------::::-------=======----------:==*:..............................................
....................................:::::-=*+:------------------::----:::.::----::----:::::::::--:::--=-..............................................
..................................::::::::---::-----::::-----::::-----::..::::--::::::::::::::::-:::==-:..............................................
.................................:::::::::----:-----:::::::::::::----:::..::::---:::::::::::::----::=--:..............................................
.................................:::::::::--=-:------::::::::::--==---::..::::----::::::::::::-----:--::..............................................
................................::::::::::----:----------::::::-==----::::::--::----::::::::-------:--::..............................................
................................:::::::::::---:------------::---==+**=======++=--------::----------:-::...............................................
................................:::::::::::---------------------=++*##+++++*#**+=------------------:::................................................
...............................::::::::::::----:---====----------==++==++++=====-::---------------::..................................................
...............................::::::::::::----:---=====-------------=====--::--:::--------===----::..................................................
..............................:::::::::::::----:-----====--------------:--:::::------------=-----::...................................................
.............................:::::::::::::::----:--------------=---======--=====--===------------::...................................................
............................::::::::::::::::---------------===+*************+++*****+=-----------::...................................................
............................:.:::::::::::::::-------=====--=========-------:--------====---------:....................................................
.............................:::::::::::::::::-------========------=============------=======---:.....................................................
............................:::::::::::::::::::------=======-------===++++++====-------=======-:......................................................
.............................:::::::::::::::::::------=======-------=========---------=======-:.......................................................
............................::::::::::::::::::::-------======----------::::----------=======--........................................................
.............................::::::::::::::::::::-------======--------------::-------======--:........................................................
.............................::::::::::::::::::::::-----===========-----==--------====++==---:........................................................
..............................::::::::::::::::::::-------====+++++++=++++=+++++=+++++++===--::........................................................
..............................::::::::::::::::::::---------===+++++++++++*****++++++++==---::::.......................................................
..............................:::::::::::::::::::::----------====+++++++++**++++++++==----::::-:......................................................
...............................:::::::::::::::::::+*-------------=====++++++++++===-----::::::=#==-:..................................................
..............................:::::::::::::::-=+*#%+--::--------------------=---------:::::::::%%#%%#*=:..............................................
.............................:::::::::::::-+%%@%%%%#--:::::::::---------------------:::::::::::#%%%%%%%#%#*+=-:.......................................
............................:::::::::-=*#%%%@@@@%%%%=-::::::::::-----------------::::::::::::::#%%%%%%%%%%%%%%%#*=-:..................................
.........................:::::::::=+#%%%@%@%@@@@@@@@%=::::::::::::-------------:::::::::::::::-#%%%%%%%%%%%@@%%%%%##*=-:..............................
.....................::::::-=+*###%%%%%%@@@%%@@@@@@@%%=:::::::::::::::-:-----:::::::::..::::::=%%%%%%%%%%%%%%@@%%#%%%###*+-:..........................
.................::::::-*#####%%%%%%%%%%%@@@%@@@@@@@@@%+-::::::::::::::::::::::::::::::::::::-*%%%%%%%%%%%%%%%@@%%%%#%%%%###*+=-:.....................
................::::::=####%%%%%%%%@%%%%%%%%%%@@@@@@@@@%*-::::::::::::::::::::::::::.:::::::-=%%%%%%%%%@%%%%%%%%%%%%%%#%%%%%####*+=:..................
.............::-:--=*#####%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@%+-::::::::::::::::::::::...::::::-+%%%%%%%%%%@@%%%%%%%%%%%%%%%%%%%%%%#####*+-:..............
..........:=+#############%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@%*=-::......:::::::::::....:::::-*%%%%%%%%%%%@@%%%%%%%%%%%%%%%%%%%%%%%%%%#####*+=::.........
.........:**##%%#########%%%%%%%%%%%%%%%%%%%%%%%@@@@@%@@@@@@%*=:::.......:::::......::::-+%%@%%%%%%%%%%%@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%######*+-:......
.......:-**####%%%%%%##%%%@@%%%%%%%%%%%%%%%%@%%%%%@@@@%%%@@@@@@#+=-:...............::-+#%@@@%%%%%%%%%%%%%@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#######*+-....
..:--=+**#%%%%%#%%%%%%%%%@@@%%%%%%%%%@%%%%%@@@%%%@@@%%%@@%@@@@@@@@@%%#***++++++++**#%%@@@@@%%%%%%%%%%%%%%@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#######**-:.
+***#%####%%%%%%%%%%%%%%@@@%%%%%%%%%%%%%%%%@@@%%%@@@%@%%%@@@%@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%#####*+-
#*##%%%##%%%%%%%%%%%%%%%@@%%%%%%%%%%%@%%%%%@@@%%%@@@%%%%%%@@@@%%%%@@@@@@@@%@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%######*
###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@%%%%%@@%%%@%%%%%%%%%@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%######
###%%%%%%%%%%%%%@@%%%%%%%%%%%%%%%%%%%%%%%%%%%@@%%%%%%%%%%%%%%@@@%@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%##
##%#%%%%%%%%%%%%@@%%%%%%%%%%%%%%%%%%%%%%%%%%%@@%%%@%%%%%%%%%%%@@@%%%%%@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#%%##
###%%%%%%%%%%%%%%@%%%%%%%%%%%%@@%%%%%%%%%%%%%%@@%%%%%%%%%%%%%%%@@@%%%%%%%%%@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = cast(Buffer, epilogue_nodes[-1])

        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=self.input_reorder)
        return code

def tuned_mm(mat1, mat2, * ,layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    llvm_template = LLVMGemmTemplate([mat1, mat2], layout)

    return llvm_template.generate().output_node()

lowerings.update({getattr(aten.mm, overload): tuned_mm for overload in aten.mm.overloads()})