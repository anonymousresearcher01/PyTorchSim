import torch
import torch.nn as nn
import torch._dynamo
import torch.utils.cpp_extension
import torch.nn.utils.prune as prune

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    message = f"|{name} Test Passed|"
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

class MLP(nn.Module):
    def __init__(self, input_size=16, hidden_size=16, output_size=16, sparsity_fc1=0, sparsity_fc2=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

        prune.l1_unstructured(self.fc1, name="weight", amount=sparsity_fc1)
        prune.l1_unstructured(self.fc2, name="weight", amount=sparsity_fc2)

        prune.remove(self.fc1, "weight")
        prune.remove(self.fc2, "weight")

    def forward(self, x):
        x = torch.sparse.mm(x, self.fc1.weight.T)
        x = torch.sparse.mm(x, self.fc2.weight.T)
        return x

def test_sparse_mlp(device, batch_size=32, input_size=128, hidden_size=128, output_size=128, bias_shift=0):
    torch.manual_seed(0)
    mlp = MLP(input_size, hidden_size, output_size, bias_shift)
    mlp = mlp.to(device=device)
    input = torch.randn(batch_size, input_size)
    x1 = input.to(device=device)
    opt_fn = torch.compile(dynamic=False)(mlp)
    res = opt_fn(x1)


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/root/workspace/PyTorchSim'))
    from Scheduler.scheduler import ExecutionEngine

    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_sparse_mlp(device)
