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

def test_reduce_sum(device, size, dim, keepdim=False):
    def reduce_sum(a, b, dim, keepdim):
        return torch.sum(a + b, axis=dim, keepdim=keepdim)
    x = torch.randn(size).to(device=device)
    y = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(reduce_sum)
    res = opt_fn(x, y, dim, keepdim)
    out = reduce_sum(x.cpu(), y.cpu(), dim, keepdim)
    test_result("ReduceSum", res, out)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_reduce_sum(device, (29, 47), 1, keepdim=True)
    test_reduce_sum(device, (17, 68), 0, keepdim=True)
    test_reduce_sum(device, (327, 447), 1, keepdim=True)
    test_reduce_sum(device, (327, 447), 0, keepdim=True)

