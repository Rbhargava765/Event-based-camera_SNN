import torch
import numpy as np
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.abspath('.'))

from obstacle_avoidance_controller import SectorizedSpikeAccumulator

print("Testing the SectorizedSpikeAccumulator with the ORIGINAL SNN model (480x640)...")

# Create the accumulator on CPU to avoid CUDA issues if any, but model will run on its device
accumulator = SectorizedSpikeAccumulator(
    num_sectors=8,
    threshold=0.5,
    decay_rate=0.8,
    device='cpu' # Accumulator logic on CPU, model defined device (e.g. CUDA) used internally by accumulator
)
print("Accumulator created successfully!")

# Generate a random event tensor with 21 time steps and 480x640 resolution
event_tensor = torch.randn(1, 2, 21, 480, 640)
print(f"Created sample event tensor with shape: {event_tensor.shape}")

print("Processing sample data...")
try:
    result = accumulator.process_event_frame(event_tensor)
    print("Sample data processed successfully!")
    print(f"Generated {len(result['sector_counts'])} sector counts")
    print(f"Generated avoidance commands: {list(result['avoidance_commands'].keys())}")
    print("Full system test with original model successful!")
except Exception as e:
    print(f"Error processing sample data: {e}")
    import traceback
    traceback.print_exc() 