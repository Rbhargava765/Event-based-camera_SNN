import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import cv2
from spikingjelly.activation_based import functional

# Add project root to path
sys.path.append(os.path.abspath('.'))  # Adjust as needed for your folder structure

# Import the patched network
from network_3d_patched.poolingNet_cat_1res import NeuronPool_Separable_Pool3d


def flow_to_color(flow_x, flow_y, max_magnitude=None):
    """
    Convert optical flow vectors to HSV color representation (then to RGB for display)
    """
    # Calculate flow magnitude and angle
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    
    # Normalize magnitude
    if max_magnitude is None:
        max_magnitude = np.max(magnitude)
    
    if max_magnitude > 0:
        normalized_magnitude = np.minimum(magnitude / max_magnitude, 1.0)
    else:
        normalized_magnitude = np.zeros_like(magnitude)
    
    # Convert angle to hue (0-360 degrees mapped to 0-1)
    angle = np.arctan2(flow_y, flow_x) / (2 * np.pi) + 0.5
    
    # Create HSV image
    hsv = np.zeros((flow_x.shape[0], flow_x.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = angle          # Hue (flow direction)
    hsv[..., 1] = 1.0            # Saturation (always 1)
    hsv[..., 2] = normalized_magnitude  # Value (flow magnitude)
    
    # Convert to RGB
    rgb = hsv_to_rgb(hsv)
    return (rgb * 255).astype(np.uint8)


def compute_epe(pred_flow, gt_flow):
    """
    Compute End Point Error between predicted and ground truth flow
    """
    # Extract x and y components
    pred_x, pred_y = pred_flow[0], pred_flow[1]
    gt_x, gt_y = gt_flow[0], gt_flow[1]
    
    # Calculate Euclidean distance at each pixel
    epe = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    mean_epe = np.mean(epe)
    
    return epe, mean_epe


def create_test_event_tensor():
    """
    Create a test event tensor with synthetic events for testing
    when the real dataset is not available
    """
    # Create a tensor of shape [1, 2, T, H, W]
    # Note: The network requires at least 21 time steps, 480x640 resolution
    T, H, W = 21, 480, 640
    events_tensor = torch.zeros(1, 2, T, H, W)
    
    # Add some synthetic events for testing
    for t in range(T):
        # Create a moving pattern
        center_x = int(W/2 + 50 * np.sin(t/T * 2 * np.pi))
        center_y = int(H/2 + 50 * np.cos(t/T * 2 * np.pi))
        
        # Add positive polarity events in a circle
        for i in range(-20, 21):
            for j in range(-20, 21):
                if i*i + j*j <= 400:  # Circle with radius 20
                    y, x = center_y + i, center_x + j
                    if 0 <= y < H and 0 <= x < W:
                        events_tensor[0, 1, t, y, x] = 1.0
        
        # Add negative polarity events in another position
        center_x2 = int(W/2 - 50 * np.sin(t/T * 2 * np.pi))
        center_y2 = int(H/2 - 50 * np.cos(t/T * 2 * np.pi))
        for i in range(-20, 21):
            for j in range(-20, 21):
                if i*i + j*j <= 400:  # Circle with radius 20
                    y, x = center_y2 + i, center_x2 + j
                    if 0 <= y < H and 0 <= x < W:
                        events_tensor[0, 0, t, y, x] = 1.0
    
    return events_tensor


def create_synthetic_ground_truth(H=480, W=640):
    """
    Create synthetic ground truth flow for testing
    """
    gt_flow_x = np.zeros((H, W))
    gt_flow_y = np.zeros((H, W))
    
    # Create a simple flow field (e.g., rotation)
    center_x, center_y = W/2, H/2
    for y in range(H):
        for x in range(W):
            # Vector from center
            dx, dy = x - center_x, y - center_y
            # Rotate 90 degrees
            gt_flow_x[y, x] = -dy / 10
            gt_flow_y[y, x] = dx / 10
    
    return gt_flow_x, gt_flow_y


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = NeuronPool_Separable_Pool3d(multiply_factor=35.0).to(device)
    
    # Try to load pre-trained weights
    model_path = 'OF_EV_SNN-main/examples/checkpoint_epoch34.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Warning: Could not find model weights at {model_path}")
    
    model.eval()
    
    # Check if we have proper DSEC lite data
    dsec_path = 'dsec_dataset_lite/data/saved_flow_data/test/event_tensors/11frames/left'
    if os.path.exists(dsec_path):
        print(f"Found DSEC lite data at {dsec_path}")
        # TODO: Load real data from DSEC lite dataset
        # This would be implemented once we have the dataset structure confirmed
        print("Real data loading not implemented yet. Using synthetic test data instead.")
        events_tensor = create_test_event_tensor()
        gt_flow_x, gt_flow_y = create_synthetic_ground_truth()
    else:
        print("DSEC lite data not found. Using synthetic test data.")
        events_tensor = create_test_event_tensor()
        gt_flow_x, gt_flow_y = create_synthetic_ground_truth()
    
    # Move tensor to device
    events_tensor = events_tensor.to(device)
    
    try:
        # Process through the model
        with torch.no_grad():
            functional.reset_net(model)
            pred_flows = model(events_tensor)
            
            # The model returns a list of predictions at different scales
            # We take the last one which should be full resolution
            pred_flow = pred_flows[-1].cpu().numpy()[0]  # Shape [2, H, W]
        
        # Extract flow components
        pred_flow_x, pred_flow_y = pred_flow[0], pred_flow[1]
        
        # Calculate end-point error
        epe, mean_epe = compute_epe((pred_flow_x, pred_flow_y), (gt_flow_x, gt_flow_y))
        
        print(f"Mean End-Point Error: {mean_epe:.4f} pixels")
        print("Note: This EPE is based on synthetic ground truth and is just for testing the pipeline.")
        if mean_epe < 3.0:
            print("✅ PASS: Mean EPE is below threshold (3.0)")
        else:
            print("❌ FAIL: Mean EPE is above threshold (3.0)")
        
        # Store test result
        test_passed = mean_epe < 3.0
        
        # Visualization - wrapped in try-except so visualization failures don't affect the test result
        try:
            # Create a simple 2D visualization of the event data
            # Sum across time and polarity dimensions to get a 2D representation
            event_vis = torch.sum(events_tensor[0], dim=(0, 1)).cpu().numpy()  # Sum across time and polarity
            event_vis = (event_vis / (event_vis.max() + 1e-6) * 255).astype(np.uint8)
            
            # Create a single output image with three panels side by side
            H, W = 480, 640
            output_img = np.zeros((H, W*3, 3), dtype=np.uint8)
            
            # Panel 1: Event visualization
            output_img[:, 0:W, 0] = event_vis  # Red channel
            output_img[:, 0:W, 1] = event_vis  # Green channel
            output_img[:, 0:W, 2] = event_vis  # Blue channel
            
            # Panel 2: Ground truth flow
            gt_flow_rgb = flow_to_color(gt_flow_x, gt_flow_y)
            output_img[:, W:2*W, :] = gt_flow_rgb
            
            # Panel 3: Predicted flow
            pred_flow_rgb = flow_to_color(pred_flow_x, pred_flow_y)
            output_img[:, 2*W:3*W, :] = pred_flow_rgb
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(output_img, "Event Visualization", (W//2-100, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(output_img, "Ground Truth Flow", (W+W//2-100, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(output_img, f"Predicted Flow (EPE: {mean_epe:.2f})", (2*W+W//2-120, 30), font, 0.7, (255, 255, 255), 2)
            
            # Save the visualization
            output_path = "optical_flow_test_results.png"
            cv2.imwrite(output_path, output_img)
            print(f"Saved visualization to {output_path}")
        except Exception as viz_error:
            print(f"Warning: Failed to generate visualization: {viz_error}")
        
        return test_passed  # Return True if test passes
        
    except Exception as e:
        print(f"Error testing optical flow: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = main()
    print(f"Test {'passed' if result else 'failed'}") 