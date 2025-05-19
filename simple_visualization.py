import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from simplified_model import SimplifiedSNNModel

def generate_test_data(flow_direction=None, magnitude=1.0, shape=(256, 256)):
    """Generate test event data with a specific flow direction"""
    height, width = shape
    
    # If no direction specified, create random direction
    if flow_direction is None:
        angle = np.random.uniform(0, 2*np.pi)
        flow_direction = [np.cos(angle), np.sin(angle)]
    
    # Normalize the direction
    flow_norm = np.sqrt(flow_direction[0]**2 + flow_direction[1]**2)
    flow_direction = [flow_direction[0]/flow_norm, flow_direction[1]/flow_norm]
    
    # Create directional flow tensor (x component)
    flow_x = np.ones((height, width)) * flow_direction[0] * magnitude
    
    # Create directional flow tensor (y component)
    flow_y = np.ones((height, width)) * flow_direction[1] * magnitude
    
    # Add some noise
    flow_x += np.random.normal(0, 0.1, (height, width))
    flow_y += np.random.normal(0, 0.1, (height, width))
    
    # Create a time sequence tensor (11 time steps as expected by the model)
    event_tensor = torch.zeros(1, 2, 11, height, width)
    
    # Fill with flow data
    for t in range(11):
        event_tensor[0, 0, t] = torch.from_numpy(flow_x).float()
        event_tensor[0, 1, t] = torch.from_numpy(flow_y).float()
    
    return event_tensor, flow_direction

def visualize_flow(flow_x, flow_y):
    """Visualize optical flow using colors"""
    # Make sure we're working with numpy arrays, not PyTorch tensors
    if isinstance(flow_x, torch.Tensor):
        flow_x = flow_x.detach().cpu().numpy()
    if isinstance(flow_y, torch.Tensor):
        flow_y = flow_y.detach().cpu().numpy()
    
    # Convert flow to HSV
    h, w = flow_x.shape
    
    # Calculate magnitude and angle
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    angle = np.arctan2(flow_y, flow_x) + np.pi
    
    # Normalize magnitude for better visualization
    max_mag = magnitude.max()
    if max_mag > 0:
        magnitude_norm = np.clip(magnitude / max_mag, 0, 1)
    else:
        magnitude_norm = magnitude
    
    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / (2 * np.pi)  # Hue (angle)
    hsv[..., 1] = 255  # Saturation (always maximum)
    hsv[..., 2] = np.clip(magnitude_norm * 255, 0, 255).astype(np.uint8)  # Value (magnitude)
    
    # Convert to RGB
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return flow_rgb

def plot_results(flow_x, flow_y, flow_direction, predicted_flow):
    """Plot the results of optical flow estimation"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert tensors to numpy arrays if needed
    if isinstance(predicted_flow, torch.Tensor):
        predicted_flow = predicted_flow.detach().cpu().numpy()
    
    # Plot input flow
    flow_rgb = visualize_flow(flow_x, flow_y)
    axs[0].imshow(flow_rgb)
    axs[0].set_title('Input Flow Pattern')
    axs[0].axis('off')
    
    # Plot output flow
    output_rgb = visualize_flow(predicted_flow[0, 0], predicted_flow[0, 1])
    axs[1].imshow(output_rgb)
    axs[1].set_title('Predicted Flow')
    axs[1].axis('off')
    
    # Plot flow arrows
    step = 20
    y, x = np.mgrid[step//2:256:step, step//2:256:step]
    
    axs[2].quiver(x, y, 
                predicted_flow[0, 0, y, x], 
                -predicted_flow[0, 1, y, x],  # Negate y for display
                color='r', scale=50, scale_units='inches')
    
    input_angle = np.arctan2(flow_direction[1], flow_direction[0])
    avg_x = np.mean(predicted_flow[0, 0])
    avg_y = np.mean(predicted_flow[0, 1])
    output_angle = np.arctan2(avg_y, avg_x)
    
    axs[2].set_title(f'Flow Direction\nInput: {input_angle:.2f} rad, Output: {output_angle:.2f} rad')
    axs[2].set_xlim(0, 255)
    axs[2].set_ylim(255, 0)  # Reverse y-axis for display
    axs[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('optical_flow_test_results.png')
    plt.show()

def test_different_flow_directions():
    """Test the model with different flow directions and create a visualization grid"""
    print("Testing the model with different flow directions...")
    
    # Create the model once
    model = SimplifiedSNNModel(multiply_factor=5.0)
    model.eval()
    
    # Define test angles (8 directions covering a full circle)
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    
    # Prepare figure for grid of results
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()
    
    # Process each direction
    for i, angle in enumerate(angles):
        flow_direction = [np.cos(angle), np.sin(angle)]
        
        # Generate test data for this direction
        event_tensor, _ = generate_test_data(flow_direction=flow_direction)
        
        # Process through model
        with torch.no_grad():
            output = model(event_tensor)
        
        # Convert to numpy
        output_np = output.detach().cpu().numpy()
        
        # Calculate average output direction
        avg_x = np.mean(output_np[0, 0])
        avg_y = np.mean(output_np[0, 1])
        output_angle = np.arctan2(avg_y, avg_x)
        
        # Visualize in the grid
        output_rgb = visualize_flow(output_np[0, 0], output_np[0, 1])
        axs[i].imshow(output_rgb)
        axs[i].set_title(f'Input: {angle:.1f} rad\nOutput: {output_angle:.1f} rad')
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('flow_direction_test_grid.png')
    plt.show()
    print("Direction test grid saved to 'flow_direction_test_grid.png'")

def main():
    print("Testing the simplified SNN model with visualization...")
    
    # Generate test data with known flow direction
    angle = np.pi / 4  # 45 degrees
    flow_direction = [np.cos(angle), np.sin(angle)]
    print(f"Generating test data with flow direction: {angle:.2f} rad ({angle * 180 / np.pi:.1f}°)")
    
    event_tensor, flow_direction = generate_test_data(flow_direction=flow_direction, magnitude=1.0)
    
    # Create the model
    model = SimplifiedSNNModel(multiply_factor=5.0)
    model.eval()
    
    # Process through the model
    with torch.no_grad():
        output = model(event_tensor)
    
    # Sample input flow for visualization
    input_flow_x = event_tensor[0, 0, 0].numpy()
    input_flow_y = event_tensor[0, 1, 0].numpy()
    
    # Visualize
    plot_results(input_flow_x, input_flow_y, flow_direction, output)
    print("Visualization saved to 'optical_flow_test_results.png'")
    
    # Test with different flow directions
    test_different_flow_directions()

if __name__ == "__main__":
    main() 