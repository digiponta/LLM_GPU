# check_gpu.py
#
# Verify that PyTorch can access a physical CUDA GPU.

import torch


def main():
    print()
    print("====================================")
    print(" LLM_GPU CUDA Check")
    print("====================================")
    print()
    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("Device          : cpu")
        print()
        print("CUDA is not available to PyTorch.")
        print("Install a CUDA-enabled PyTorch build and verify the NVIDIA driver.")
        return

    device = torch.device("cuda")
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)

    print("Device          :", device)
    print("GPU index       :", index)
    print("GPU              :", torch.cuda.get_device_name(index))
    print("VRAM             :", f"{props.total_memory / (1024 ** 3):.2f} GiB")
    print("CUDA runtime     :", torch.version.cuda)

    # Tiny real GPU computation.
    a = torch.randn((1024, 1024), device=device)
    b = torch.randn((1024, 1024), device=device)
    c = a @ b
    torch.cuda.synchronize()

    print("GPU matmul       : OK")
    print("Result device    :", c.device)


if __name__ == "__main__":
    main()
