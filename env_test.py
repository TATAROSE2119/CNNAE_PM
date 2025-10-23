import torch


def check_torch_gpu():
    print("========== PyTorch 环境检测 ==========")

    # 1. 检查 PyTorch 是否安装
    try:
        print(f"✅ PyTorch 版本：{torch.__version__}")
    except Exception as e:
        print("❌ 未检测到 PyTorch，请先安装：pip install torch")
        return

    # 2. 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"💡 CUDA 可用：{cuda_available}")

    if not cuda_available:
        # 提供排查建议
        print("⚠️ 检测到 CUDA 不可用，可能原因：")
        print("  1. 未安装 CUDA Toolkit 或版本不匹配。")
        print("  2. GPU 驱动未安装或版本过低。")
        print("  3. 当前 PyTorch 是 CPU 版本（需安装带 cuda 的版本）。")
        print("  4. 使用的是虚拟机/WSL 未启用 GPU 直通。")
        return

    # 3. 检查 GPU 信息
    device_count = torch.cuda.device_count()
    print(f"🖥️ 检测到 GPU 数量：{device_count}")
    for i in range(device_count):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        print(f"     当前显存占用：{torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        print(f"     可用显存：{torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")

    # 4. 检查当前设备
    current_device = torch.cuda.current_device()
    print(f"🎯 当前默认设备：{torch.cuda.get_device_name(current_device)}")
    print("=====================================")


if __name__ == "__main__":
    check_torch_gpu()
