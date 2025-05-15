import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test CUDA with a simple operation
    x = torch.rand(5, 5).cuda()
    print(f"Tensor on GPU: {x.device}")
    print("CUDA test successful!") 