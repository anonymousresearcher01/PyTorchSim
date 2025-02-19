import torch
import torch.nn as nn
import torch._dynamo
import torch.utils.cpp_extension

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
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.relu = nn.ReLU()

        # bias_mean = -0.7
        # bias_std = 0.5
        # self.fc1.bias.data = torch.normal(mean=bias_mean, std=bias_std, size=self.fc1.bias.shape)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.relu(x)
        x = torch.sparse.mm(x, self.fc2.weight.T) + self.fc2.bias
        return x

def test_sparse_mlp(device, batch=32, input_size=128, hidden_size=128, output_size=128):
    torch.manual_seed(5462)
    mlp = MLP(input_size, hidden_size, output_size)
    mlp = mlp.to(device=device)
    input = torch.randn(batch, input_size)
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
    test_sparse_mlp(device, 32, 128, 128, 128)
