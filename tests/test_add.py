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

def test_vectoradd(device, size=(128, 128)):
    def vectoradd(a, b):
        return a + b
    x = torch.randn(size).to(device=device)
    y = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vectoradd)
    res = opt_fn(x, y)
    out = vectoradd(x.cpu(), y.cpu())
    test_result("VectorAdd", res, out)

def test_vector_scalar_add(device, size=(128, 128)):
    def vectoradd(a, b):
        return a + b
    x = torch.randn(size).to(device=device)
    y = torch.randn([1]).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vectoradd)
    res = opt_fn(x, y)
    out = vectoradd(x.cpu(), y.cpu())
    test_result("VectorScalarAdd", res, out)


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_vectoradd(device, (47, 10))
    test_vectoradd(device, (128, 128))
    test_vectoradd(device, (4071, 429))
