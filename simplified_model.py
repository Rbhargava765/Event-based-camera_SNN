import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, base

class MultiplyBy(nn.Module):
    """Simple multiplication layer for scaling inputs"""
    def __init__(self, scale_value=5.0):
        super(MultiplyBy, self).__init__()
        self.scale_value = scale_value

    def forward(self, input):
        return torch.mul(input, self.scale_value)

# Make sure our model inherits from base.MemoryModule to support reset_net
class SimplifiedSNNModel(base.MemoryModule):
    """
    A simplified SNN model for optical flow estimation from event data.
    This model is designed to work with 256x256 input images.
    """
    def __init__(self, multiply_factor=5.0):
        super(SimplifiedSNNModel, self).__init__()
        
        print("USING SIMPLIFIED SPIKING MODEL")
        
        # Encoder pathway
        self.encoder = nn.Sequential(
            # Initial 3D convolution [batch, 2, time_steps, height, width]
            nn.Conv3d(in_channels=2, out_channels=16, kernel_size=(5, 3, 3), 
                      stride=1, padding=(0, 1, 1), bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(),
            
            # Max pooling to reduce spatial dimensions
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Middle processing
        self.middle = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, 
                     stride=1, padding=1, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                     stride=1, padding=1, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode()
        )
        
        # Decoder/output pathway
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,
                     stride=1, padding=1, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(),
            
            # Final layer for optical flow (2 channels: x and y)
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3,
                     stride=1, padding=1, bias=False),
            
            # No activation for the final output (direct flow values)
        )
        
        # For accumulating neuron outputs
        self.pool = neuron.IFNode(v_threshold=float('inf'), v_reset=0.)
    
    def forward(self, x):
        """
        Forward pass of the simplified SNN model.
        
        Args:
            x: Input tensor of shape [batch, channels, time_steps, height, width]
               Expected dimensions: [1, 2, 11, 256, 256]
        
        Returns:
            Optical flow prediction of shape [batch, 2, height, width]
        """
        # Process through 3D encoder
        encoded = self.encoder(x)
        
        # Take final time step and remove time dimension
        time_collapsed = encoded[:, :, -1]  # Shape: [batch, 16, height/2, width/2]
        
        # Process through 2D middle layers
        middle_output = self.middle(time_collapsed)  # Shape: [batch, 32, height/2, width/2]
        
        # Upsample back to original resolution
        upsampled = nn.functional.interpolate(
            middle_output,
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )
        
        # Process through decoder for final optical flow prediction
        decoded = self.decoder(upsampled)  # Shape: [batch, 2, 256, 256]
        
        # Accumulate spikes for final output
        self.pool(decoded)
        flow_prediction = self.pool.v
        
        # Return just the final prediction
        return flow_prediction
    
    def reset(self):
        """Reset the states of all spiking neurons in the model"""
        functional.reset_net(self)

def test_model():
    """Test the simplified model with random input"""
    # Create random input tensor [batch, channels, time_steps, height, width]
    x = torch.randn(1, 2, 11, 256, 256)
    
    # Initialize model
    model = SimplifiedSNNModel()
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Print output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    return output

if __name__ == "__main__":
    test_model() 