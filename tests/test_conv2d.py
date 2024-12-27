import torch
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

def test_conv2d(device):
    def custom_conv2d(a, b):
        i_c = a.shape[1]
        o_c = b.shape[0]
        conv2d = torch.nn.Conv2d(i_c, o_c, b.shape[-1], stride=1, padding=0, dilation=1)
        conv2d.weight = torch.nn.Parameter(b)
        return conv2d(a)
    torch.manual_seed(0)
    conv_input = torch.randn(1, 8, 64, 64).to(device=device)
    conv_kernel = torch.randn(16, 8, 3, 3).to(device=device)
    opt_fn = torch.compile(dynamic=False)(custom_conv2d)
    res = opt_fn(conv_input, conv_kernel)
    out = custom_conv2d(conv_input.cpu(), conv_kernel.cpu())
    test_result("Conv2d Forward", res, out, rtol=1e-1, atol=1e-1)
    print("Max diff > ", torch.max(torch.abs(res.cpu() - out)))

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath("/workspace/PyTorchSim"))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_conv2d(device)
