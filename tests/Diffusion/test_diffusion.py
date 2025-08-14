import os
import sys
import math
import argparse
import torch
import torch._dynamo
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D
from diffusers.models.resnet import ResnetBlock2D

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        diff = torch.max(torch.abs(out.cpu() - cpu_out)).item()
        print(f"Max abs diff: {diff}")
        exit(1)

@torch.no_grad()
def test_unet_conditional(
    device,
    model_id="runwayml/stable-diffusion-v1-5",
    batch=1,
    dtype="float32",
    rtol=1e-4,
    atol=1e-4,
    prompt="a cat in a hat",
):
    from diffusers import DiffusionPipeline

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    print(f"Loading pipeline: {model_id} (dtype={torch_dtype})")
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe.to("cpu")

    # UNet/구성 정보
    unet = pipe.unet.eval()
    in_ch = unet.config.in_channels
    latent_sz = getattr(unet.config, "sample_size", 64)
    cross_dim = getattr(unet.config, "cross_attention_dim", None)

    # 입력(latents) 생성: [B, C, H, W]
    g = torch.Generator().manual_seed(0)
    latents = torch.randn(batch, in_ch, latent_sz, latent_sz, generator=g, dtype=torch_dtype)

    # timestep (스칼라 또는 [B]) — UNet은 보통 float/long 모두 허용
    timestep = torch.tensor(999, dtype=torch.float32)

    # encoder_hidden_states 준비
    enc_states_cpu = None
    # 1) tokenizer + text_encoder 사용 가능하면 실제 임베딩 사용
    if hasattr(pipe, "tokenizer") and hasattr(pipe, "text_encoder") and cross_dim is not None:
        try:
            tokens = pipe.tokenizer(
                [prompt] * batch,
                padding="max_length",
                max_length=getattr(pipe.tokenizer, "model_max_length", 77),
                truncation=True,
                return_tensors="pt",
            )
            text_out = pipe.text_encoder(input_ids=tokens.input_ids).last_hidden_state  # [B, T, D]
            if text_out.shape[-1] != cross_dim:
                # 크기가 맞지 않는 특수 파이프라인은 랜덤으로 폴백
                print(f"Warning: text_encoder dim {text_out.shape[-1]} != cross_attn dim {cross_dim}. Fallback to random.")
                raise RuntimeError("cross-dim mismatch")
            enc_states_cpu = text_out.to(dtype=torch_dtype)
        except Exception as e:
            print(f"Text encoder unavailable or mismatch: {e}. Fallback to random encoder states.")
    # 2) 폴백: 랜덤 임베딩 (cross_attention_dim이 정의된 경우)
    if enc_states_cpu is None:
        if cross_dim is None:
            # cross-attn이 없는 UNet이면 None 전달 (일부 모델)
            enc_states_cpu = None
        else:
            seq_len = 77  # 일반적인 텍스트 토큰 길이
            enc_states_cpu = torch.randn(batch, seq_len, cross_dim, generator=g, dtype=torch_dtype)

    latents_dev = latents.to(device)
    timestep_dev = timestep.to(device)
    if enc_states_cpu is not None:
        enc_states_dev = enc_states_cpu.to(device)
    else:
        enc_states_dev = None

    print("Compiling UNet with torch.compile(...)")
    unet_dev = unet.to(device)
    unet_compiled = torch.compile(unet_dev, dynamic=False)

    # Forward (device)
    with torch.no_grad():
        if enc_states_dev is None:
            out_dev = unet_compiled(latents_dev, timestep_dev).sample
        else:
            out_dev = unet_compiled(latents_dev, timestep_dev, encoder_hidden_states=enc_states_dev).sample

        unet_cpu = unet.to("cpu")
        if enc_states_cpu is None:
            out_cpu = unet_cpu(latents.cpu(), timestep).sample
        else:
            out_cpu = unet_cpu(latents.cpu(), timestep, encoder_hidden_states=enc_states_cpu).sample

    test_result(f"UNet({model_id}) forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(out_dev.cpu() - out_cpu)).item())
    print("UNet Simulation Done")

def test_cross_attn_down_block2d(
    device,
    in_channels=320,
    out_channels=320,
    temb_channels=1280,
    cross_attention_dim=768,
    batch=1,
    height=32,
    width=32,
    rtol=1e-4,
    atol=1e-4,
    num_layers=1,
    num_attention_heads=8,
    dual_cross_attention=False
):
    print(f"Testing CrossAttnDownBlock2D on device: {device}")
    
    # 1. Initialize the module on CPU
    cpu_block = CrossAttnDownBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        num_layers=num_layers,
        cross_attention_dim=cross_attention_dim,
        num_attention_heads=num_attention_heads,
        dual_cross_attention=dual_cross_attention
    ).to("cpu").eval()

    # 2. Create synthetic inputs on CPU
    g = torch.Generator().manual_seed(0)
    hidden_states_cpu = torch.randn(batch, in_channels, height, width, generator=g)
    temb_cpu = torch.randn(batch, temb_channels, generator=g)
    encoder_hidden_states_cpu = torch.randn(batch, 77, cross_attention_dim, generator=g)

    # 3. Get the output from the CPU module
    with torch.no_grad():
        cpu_out, _ = cpu_block(
            hidden_states=hidden_states_cpu,
            temb=temb_cpu,
            encoder_hidden_states=encoder_hidden_states_cpu,
        )
    
    # 4. Initialize the module on the custom device
    device_block = cpu_block.to(device).eval()
    device_block = torch.compile(device_block, dynamic=False)

    # 5. Move inputs to the custom device
    hidden_states_dev = hidden_states_cpu.to(device)
    temb_dev = temb_cpu.to(device)
    encoder_hidden_states_dev = encoder_hidden_states_cpu.to(device)
    
    # 6. Get the output from the custom device module
    with torch.no_grad():
        dev_out, _ = device_block(
            hidden_states=hidden_states_dev,
            temb=temb_dev,
            encoder_hidden_states=encoder_hidden_states_dev,
        )

    # 7. Compare the results
    test_result("CrossAttnDownBlock2D", dev_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(dev_out.cpu() - cpu_out)).item())
    print("CrossAttnDownBlock2D simulation done.")

def test_resnetblock2d_down(
    device,
    batch=1,
    channels=320,          # in_channels == out_channels
    height=32,
    width=32,
    temb_channels=128,
    resnet_eps=1e-5,
    resnet_groups=32,
    dropout=0.0,
    resnet_time_scale_shift="default",   # e.g., "default" | "scale_shift"
    resnet_act_fn="swish",
    output_scale_factor=1.0,
    resnet_pre_norm=True,
    rtol=1e-4,
    atol=1e-4,
    stride=None,
):
    print(f"Testing ResnetBlock2D(down=True) on device: {device}")

    in_channels = channels
    out_channels = channels
    g = torch.Generator().manual_seed(0)

    cpu_blk = ResnetBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        eps=resnet_eps,
        groups=resnet_groups,
        dropout=dropout,
        time_embedding_norm=resnet_time_scale_shift,
        non_linearity=resnet_act_fn,
        output_scale_factor=output_scale_factor,
        pre_norm=resnet_pre_norm
    ).to("cpu").eval()

    if stride is not None:
        x_cpu = torch.empty_strided([batch, in_channels, height, width], stride).normal_()
    else:
        x_cpu = torch.randn(batch, in_channels, height, width, generator=g)

    temb_cpu = torch.randn(batch, temb_channels, generator=g)

    with torch.no_grad():
        y_cpu = cpu_blk(x_cpu, temb=temb_cpu)

    dev_blk = cpu_blk.to(device).eval()
    dev_blk = torch.compile(dev_blk, dynamic=False)

    x_dev = x_cpu.to(device)
    temb_dev = temb_cpu.to(device)

    with torch.no_grad():
        y_dev = dev_blk(x_dev, temb=temb_dev)

    try:
        test_result("ResnetBlock2D(down=True)", y_dev, y_cpu, rtol=rtol, atol=atol)
    except NameError:
        # fallback: PyTorch의 기본 엄밀 비교
        torch.testing.assert_close(y_dev.cpu(), y_cpu, rtol=rtol, atol=atol)
        print("ResnetBlock2D(down=True) close-check passed.")

    max_diff = torch.max(torch.abs(y_dev.cpu() - y_cpu)).item()
    print("Max diff >", max_diff)
    print("ResnetBlock2D(down=True) simulation done.")

def test_groupnorm(
    device,
    batch=1,
    channels=320,
    height=32,
    width=32,
    num_groups=32,
    eps=1e-5,
    rtol=1e-4,
    atol=1e-4,
    stride=None
):
    print(f"Testing GroupNorm on device: {device}")

    # 1. Initialize the module on CPU
    cpu_norm = torch.nn.GroupNorm(
        num_groups=num_groups, 
        num_channels=channels, 
        eps=eps, 
        affine=True
    ).to("cpu").eval()

    # 2. Create synthetic inputs on CPU
    g = torch.Generator().manual_seed(0)
    if stride is not None:
        input_cpu = torch.empty_strided([batch, channels, height, width], stride)
        input_cpu = input_cpu.normal_()
    else:
        input_cpu = torch.randn(batch, channels, height, width, generator=g)

    # 3. Get the output from the CPU module
    with torch.no_grad():
        cpu_out = cpu_norm(input_cpu)

    # 4. Initialize the module on the custom device
    device_norm = torch.nn.GroupNorm(
        num_groups=num_groups, 
        num_channels=channels, 
        eps=eps, 
        affine=True
    ).to(device).eval()
    device_norm = torch.compile(device_norm, dynamic=False)
    
    # Copy the weights from the CPU module to ensure they are identical
    device_norm.weight.data.copy_(cpu_norm.weight.data)
    device_norm.bias.data.copy_(cpu_norm.bias.data)

    # 5. Move inputs to the custom device
    input_dev = input_cpu.to(device)

    # 6. Get the output from the custom device module
    with torch.no_grad():
        dev_out = device_norm(input_dev)

    # 7. Compare the results
    test_result("GroupNorm", dev_out, cpu_out, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(dev_out.cpu() - cpu_out)).item())
    print("GroupNorm simulation done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNet (diffusers) test with comparison")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Diffusers model id (e.g., Qwen/Qwen-Image)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--prompt", type=str, default="a cat in a hat")
    args = parser.parse_args()

    sys.path.append(os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"))
    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()

    test_groupnorm(device)
    test_groupnorm(device, stride=[1, 1, 320*32, 320])
    test_resnetblock2d_down(device)
    #test_cross_attn_down_block2d(device)
    #test_unet_conditional(
    #    device=device,
    #    model_id=args.model,
    #    batch=args.batch,
    #    dtype=args.dtype,
    #    rtol=args.rtol,
    #    atol=args.atol,
    #    prompt=args.prompt,
    #)