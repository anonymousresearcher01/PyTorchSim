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

def test_view3D_2D(device, size=(16, 8, 16), t_x=0, t_y=1):
    def view3D_2D(a):
        return a.transpose(t_x, t_y).contiguous().view(-1, size[0] * size[2])
    torch.manual_seed(0)
    cpu_input = torch.randn(size)
    input = cpu_input.clone().to(device=device)
    opt_fn = torch.compile(dynamic=False)(view3D_2D)
    res = opt_fn(input)
    out = view3D_2D(cpu_input)
    test_result("view 3D->2D", res, out)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_view3D_2D(device)
    test_view3D_2D(device, [12, 512, 64])

