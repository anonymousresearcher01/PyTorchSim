from typing import List
import os
from torch.fx.passes.graph_drawer import FxGraphDrawer
os.environ['TORCH_LOGS'] = 'bytecode'
import torch

def dummy_compiler(gm: torch.fx.GraphModule, _):
    gm.graph.print_tabular()
    drawer = FxGraphDrawer(gm, "my_model")
    drawer.get_dot_graph().write_svg("fx_graph.svg")
    return gm.forward # Return a callable object

class MyModel(torch.nn.Module):
    def forward(self, x, y):
        z = torch.matmul(x, y)
        return torch.relu(z)

@torch.compile(backend=dummy_compiler)
def f(x, y):
    my_model = MyModel()
    return my_model(x, y)

if __name__ == "__main__":
    x = torch.randn(7, 5,requires_grad=False)
    y = torch.randn(5, 3,requires_grad=False)
    k = f(x, y)
    print(k)
