import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.abspath('.'))

from obstacle_avoidance_controller import SectorizedSpikeAccumulator

def visualize_sectors(accumulator):
    """
    Create a simple visualization of the sectors and their current activation values.
    
    Args:
        accumulator: SectorizedSpikeAccumulator instance
    """
    # Create a circular image for the visualization
    height, width = 480, 640
    vis_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw sector boundaries
    center_y, center_x = height // 2, width // 2
    radius = min(height, width) // 2 - 20
    
    # Draw each sector with appropriate color
    for i in range(accumulator.num_sectors):
        # Get sector boundaries
        sector_size = 2 * np.pi / accumulator.num_sectors
        start_angle = i * sector_size
        end_angle = (i + 1) * sector_size
        
        # Normalize accumulator value for color intensity
        max_value = max(1.0, np.max(accumulator.accumulators))
        intensity = accumulator.accumulators[i] / max_value
        
        # Calculate color (red for high values)
        color = (0, 0, int(255 * intensity)) if accumulator.accumulators[i] <= accumulator.threshold else (0, 0, 255)
        
        # Draw sector as a filled pie slice
        angles = np.linspace(start_angle, end_angle, 100)
        pts = np.array([[center_x, center_y]] + 
                      [(center_x + int(radius * np.cos(angle)), 
                         center_y + int(radius * np.sin(angle))) for angle in angles], 
                      dtype=np.int32)
        cv2.fillPoly(vis_img, [pts], color)
        
        # Add sector number label
        label_angle = (start_angle + end_angle) / 2
        label_x = int(center_x + (radius * 0.7) * np.cos(label_angle))
        label_y = int(center_y + (radius * 0.7) * np.sin(label_angle))
        cv2.putText(vis_img, str(i), (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw center and threshold indicator
    cv2.circle(vis_img, (center_x, center_y), 10, (0, 255, 0), -1)  # Green center
    cv2.circle(vis_img, (center_x, center_y), radius, (128, 128, 128), 2)  # Gray circle for reference
    
    # Add threshold text
    cv2.putText(vis_img, f"Threshold: {accumulator.threshold:.2f}", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add information about active sectors
    active_sectors = np.where(accumulator.accumulators > accumulator.threshold)[0]
    if len(active_sectors) > 0:
        cv2.putText(vis_img, f"Active Sectors: {active_sectors}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return vis_img

def generate_test_event_data(frame_size=(480, 640), flow_direction=None, magnitude=1.0):
    """
    Generate test event data with a specific flow direction.
    
    Args:
        frame_size: Size of the frame (height, width)
        flow_direction: Direction vector [x, y] or None for random
        magnitude: Flow magnitude
        
    Returns:
        torch.Tensor: Event tensor for testing
    """
    height, width = frame_size
    
    # If no direction specified, create random direction
    if flow_direction is None:
        angle = np.random.uniform(0, 2*np.pi)
        flow_direction = [np.cos(angle), np.sin(angle)]
    
    # Normalize the direction
    flow_norm = np.sqrt(flow_direction[0]**2 + flow_direction[1]**2)
    flow_direction = [flow_direction[0]/flow_norm, flow_direction[1]/flow_norm]
    
    # Create flow tensors
    flow_x = np.ones((height, width)) * flow_direction[0] * magnitude
    flow_y = np.ones((height, width)) * flow_direction[1] * magnitude
    
    # Add some noise
    flow_x += np.random.normal(0, 0.1, (height, width))
    flow_y += np.random.normal(0, 0.1, (height, width))
    
    # Create a time sequence tensor with 21 time steps
    event_tensor = torch.zeros(1, 2, 21, height, width)
    
    # Fill with flow data
    for t in range(21):
        event_tensor[0, 0, t] = torch.from_numpy(flow_x).float()
        event_tensor[0, 1, t] = torch.from_numpy(flow_y).float()
    
    return event_tensor, flow_direction

def main():
    # Create accumulator on CPU for visualization
    accumulator = SectorizedSpikeAccumulator(
        num_sectors=8,
        threshold=1.5,
        decay_rate=0.8,
        device='cpu'
    )
    
    # Set up OpenCV window
    cv2.namedWindow("Sector Visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sector Visualization", 800, 600)
    
    print("Starting visualization. Press ESC to exit.")
    print("Each time a new random flow direction will be generated.")
    
    try:
        while True:
            # Generate random flow data
            event_tensor, flow_direction = generate_test_event_data(magnitude=2.0)
            print(f"Flow direction: [{flow_direction[0]:.2f}, {flow_direction[1]:.2f}]")
            
            # Process through accumulator
            result = accumulator.process_event_frame(event_tensor)
            
            # Create visualization
            vis_img = visualize_sectors(accumulator)
            
            # Draw arrow indicating flow direction
            center_y, center_x = vis_img.shape[0] // 2, vis_img.shape[1] // 2
            arrow_length = 100
            end_x = int(center_x + arrow_length * flow_direction[0])
            end_y = int(center_y + arrow_length * flow_direction[1])
            cv2.arrowedLine(vis_img, (center_x, center_y), (end_x, end_y), 
                           (0, 255, 255), 3, tipLength=0.3)
            
            # Show the visualization
            cv2.imshow("Sector Visualization", vis_img)
            
            # Show avoidance commands
            avoidance_commands = result['avoidance_commands']
            active_commands = [cmd for cmd, value in avoidance_commands.items() if value]
            if active_commands:
                print(f"Active commands: {active_commands}")
            else:
                print("No active commands")
            
            # Wait for key press (wait for 1 second, or exit on ESC)
            key = cv2.waitKey(1000)
            if key == 27:  # ESC key
                break
            
    except KeyboardInterrupt:
        print("Exiting...")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 