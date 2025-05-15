from spikingjelly.activation_based import neuron, functional
import torch

print("SpikingJelly imported successfully")
print("Creating a simple IFNode...")

# Create a simple spiking neuron
ifnode = neuron.IFNode()

# Run a simple test
x = torch.rand(10)
out = ifnode(x)
print(f"Input: {x}")
print(f"Output: {out}")
print(f"Neuron voltage: {ifnode.v}") 