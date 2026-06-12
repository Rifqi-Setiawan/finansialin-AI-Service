import torch

def check_gpu():
    is_available = torch.cuda.is_available()
    print("="*40)
    print("GPU Check:")
    print(f"CUDA Available: {is_available}")
    
    if is_available:
        try:
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"Total VRAM: {total_mem / (1024**3):.2f} GB")
            print(f"Free VRAM: {free_mem / (1024**3):.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        except Exception as e:
            print(f"Error reading GPU info: {e}")
    else:
        print("WARNING: CUDA is not available. PyTorch will use CPU.")
        print("To fix this, ensure you installed PyTorch with CUDA support (e.g., --index-url https://download.pytorch.org/whl/cu121).")
    print("="*40)

if __name__ == "__main__":
    check_gpu()
