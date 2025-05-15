import torch
import numpy as np
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.abspath('.'))
# Add the OF_EV_SNN-main directory to the path
sys.path.append(os.path.join(os.path.abspath('.'), 'OF_EV_SNN-main'))

from obstacle_avoidance_controller import SectorizedSpikeAccumulator

print("Testing the SectorizedSpikeAccumulator...")

# Create the accumulator on CPU to avoid CUDA issues if any
accumulator = SectorizedSpikeAccumulator(
    num_sectors=8,
    threshold=0.5,
    decay_rate=0.8,
    device='cpu'
)

print("Accumulator created successfully!")

# Generate a random event tensor with more time steps (sample data)
# Using 21 time steps to match the model's requirements
event_tensor = torch.randn(1, 2, 21, 480, 640)
print(f"Created sample event tensor with shape: {event_tensor.shape}")

# Process it - this tests most of the pipeline
print("Processing sample data...")
try:
    result = accumulator.process_event_frame(event_tensor)
    print("Sample data processed successfully!")
    print(f"Generated {len(result['sector_counts'])} sector counts")
    print(f"Generated avoidance commands: {list(result['avoidance_commands'].keys())}")
    print("Full system test successful!")
except Exception as e:
    print(f"Error processing sample data: {e}") 