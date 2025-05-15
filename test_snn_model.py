import torch
import sys
import os
import traceback

print("Starting SNN model test...")

# Add the project directory to the path
sys.path.append(os.path.abspath('.'))
# Add the OF_EV_SNN-main directory to the path
sys.path.append(os.path.join(os.path.abspath('.'), 'OF_EV_SNN-main'))

print("Python paths:", sys.path)

try:
    print("Attempting to import from network_3d_patched...")
    from network_3d_patched.poolingNet_cat_1res import NeuronPool_Separable_Pool3d
    print("Successfully imported from network_3d_patched")
except ImportError as e:
    print(f"Import error from network_3d_patched: {e}")
    try:
        # Try alternative import path
        print("Attempting to import from network_3d...")
        from network_3d.poolingNet_cat_1res import NeuronPool_Separable_Pool3d
        print("Successfully imported from network_3d")
    except ImportError as e:
        print(f"Error: Could not import NeuronPool_Separable_Pool3d: {e}")
        sys.exit(1)

# Check if SEWResidual exists in spikingjelly
print("Checking spikingjelly modules...")
import spikingjelly
try:
    version = getattr(spikingjelly, '__version__', 'Unknown')
    print(f"SpikingJelly version: {version}")
except Exception as e:
    print(f"Could not get SpikingJelly version: {e}")

try:
    from spikingjelly.clock_driven import layer
    print("Available modules in spikingjelly.clock_driven.layer:", dir(layer))
    if hasattr(layer, 'SEWResidual'):
        print("SEWResidual layer is available")
    else:
        print("SEWResidual layer is NOT available")
except Exception as e:
    print(f"Error checking spikingjelly modules: {e}")

print("Creating SNN model...")
try:
    model = NeuronPool_Separable_Pool3d(multiply_factor=35.0)
    print("Model created successfully!")
except Exception as e:
    print(f"Error creating model: {e}")
    traceback.print_exc()
    sys.exit(1)

# Check for pre-trained weights
checkpoint_path = 'OF_EV_SNN-main/examples/checkpoint_epoch34.pth'
if os.path.exists(checkpoint_path):
    print(f"Found pre-trained weights at {checkpoint_path}")
    try:
        # Load on CPU to avoid CUDA errors if not available
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # Just check keys instead of loading (safer)
        print(f"Checkpoint contains {len(checkpoint.keys())} parameter sets")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        traceback.print_exc()
else:
    print(f"Pre-trained weights not found at {checkpoint_path}")

print("SNN model test completed") 