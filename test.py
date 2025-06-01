import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available (for NVIDIA/AMD ROCm via PyTorch): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Number of GPUs PyTorch sees: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    # For NVIDIA, PyTorch is often linked against a specific CUDA version
    if hasattr(torch.version, 'cuda') and torch.version.cuda:
        print(f"PyTorch CUDA version: {torch.version.cuda}")
    # For AMD (ROCm), PyTorch might report a HIP version if available
    if hasattr(torch.version, 'hip') and torch.version.hip:
        print(f"PyTorch HIP (ROCm) version: {torch.version.hip}")
else:
    print("CUDA (GPU support) is NOT available to PyTorch in this environment.")
    # Check if it's a CPU-only build
    if "cpu" in torch.__version__: # A simple heuristic
        print("This looks like a CPU-only PyTorch build.")

# For Apple Silicon (MPS) - your original code had this check
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f"MPS (Apple Silicon GPU) is available: True")
    if not torch.cuda.is_available(): # If no CUDA, but MPS is there, suggest MPS.
        print("Consider using 'mps' as your device if you are on an Apple Silicon Mac and CUDA is not expected.")
else:
    print(f"MPS (Apple Silicon GPU) is available: False")