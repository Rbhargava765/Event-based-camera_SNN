import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import torch
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.abspath('.'))

from obstacle_avoidance_controller import SectorizedSpikeAccumulator
import time
import argparse
from matplotlib.animation import FuncAnimation

def create_sector_visualization(accumulator, frame_size=(480, 640)):
    """
    Create a visualization of the sector accumulation.
    
    Args:
        accumulator: SectorizedSpikeAccumulator instance
        frame_size: Size of the frame (height, width)
        
    Returns:
        np.ndarray: Visualization image
    """
    height, width = frame_size
    vis_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a colormap (red for high values)
    colors = [(0, 0, 0), (1, 0, 0)]  # Black to red
    cmap = LinearSegmentedColormap.from_list("danger_cmap", colors)
    
    # Normalize accumulator values for color mapping
    max_value = max(1.0, np.max(accumulator.accumulators))
    normalized_values = accumulator.accumulators / max_value
    
    # Draw each sector with appropriate color intensity
    for i, mask in enumerate(accumulator.sector_masks):
        color_intensity = normalized_values[i]
        color = np.array(cmap(color_intensity)[0:3]) * 255
        
        # Create a colored mask for this sector
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]
        
        # Add to visualization
        vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, 1.0, 0)
    
    # Add a center marker
    center_y, center_x = height // 2, width // 2
    cv2.circle(vis_img, (center_x, center_y), 10, (0, 255, 0), -1)
    
    # Add sector numbers
    for i in range(accumulator.num_sectors):
        angle_rad = 2 * np.pi * i / accumulator.num_sectors
        r = min(height, width) // 3  # Radius for text placement
        text_x = int(center_x + r * np.cos(angle_rad))
        text_y = int(center_y - r * np.sin(angle_rad))  # Negative because y-axis is flipped
        cv2.putText(vis_img, str(i), (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return vis_img

def visualize_with_matplotlib(accumulator):
    """
    Create a matplotlib visualization of the accumulator sectors.
    
    Args:
        accumulator: SectorizedSpikeAccumulator instance
    """
    plt.figure(figsize=(12, 10))
    
    # Polar plot for accumulator values
    ax1 = plt.subplot(121, projection='polar')
    theta = np.linspace(0, 2*np.pi, accumulator.num_sectors, endpoint=False)
    radii = accumulator.accumulators
    
    # Plot each sector as a bar
    bars = ax1.bar(theta, radii, width=2*np.pi/accumulator.num_sectors, alpha=0.5)
    
    # Color the bars based on whether they exceed the threshold
    for i, bar in enumerate(bars):
        if accumulator.accumulators[i] > accumulator.threshold:
            bar.set_facecolor('red')
        else:
            bar.set_facecolor('blue')
    
    ax1.set_title('Sector Accumulation Values')
    ax1.set_theta_zero_location('N')  # 0 degrees at the top
    ax1.set_theta_direction(-1)  # Clockwise
    
    # Add threshold circle
    ax1.plot(np.linspace(0, 2*np.pi, 100), 
             np.ones(100) * accumulator.threshold, 
             'r--', alpha=0.7, label='Threshold')
    ax1.legend()
    
    # Regular plot for avoidance commands
    ax2 = plt.subplot(122)
    avoidance_commands = accumulator._generate_avoidance_commands()
    
    # Get command names and values
    cmd_names = list(avoidance_commands.keys())
    cmd_values = [int(v) for v in avoidance_commands.values()]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(cmd_names))
    ax2.barh(y_pos, cmd_values, align='center', 
             color=['green' if v else 'gray' for v in cmd_values])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(cmd_names)
    ax2.set_xlim(0, 1.1)
    ax2.set_title('Avoidance Commands')
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

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
    
    # Create directional flow tensor (x component)
    flow_x = np.ones((height, width)) * flow_direction[0] * magnitude
    
    # Create directional flow tensor (y component)
    flow_y = np.ones((height, width)) * flow_direction[1] * magnitude
    
    # Add some noise
    flow_x += np.random.normal(0, 0.1, (height, width))
    flow_y += np.random.normal(0, 0.1, (height, width))
    
    # Create a 21-frame time sequence tensor (updated from 11 to 21)
    event_tensor = torch.zeros(1, 2, 21, height, width)
    
    # Fill with flow data (assume same flow across all time steps)
    for t in range(21):  # Updated loop range to match 21 time steps
        event_tensor[0, 0, t] = torch.from_numpy(flow_x).float()
        event_tensor[0, 1, t] = torch.from_numpy(flow_y).float()
    
    return event_tensor

def interactive_visualization():
    """Run an interactive visualization of the sectorized accumulator"""
    # Create accumulator
    accumulator = SectorizedSpikeAccumulator(
        num_sectors=8,
        threshold=2.0,
        decay_rate=0.8,
        device='cpu'  # Use CPU for visualization
    )
    
    # Set up figure for animation
    fig = plt.figure(figsize=(12, 5))
    
    # Create polar subplot
    ax1 = fig.add_subplot(121, projection='polar')
    ax2 = fig.add_subplot(122)
    
    # Initialize polar plot
    theta = np.linspace(0, 2*np.pi, accumulator.num_sectors, endpoint=False)
    bars = ax1.bar(theta, np.zeros_like(theta), width=2*np.pi/accumulator.num_sectors, alpha=0.7)
    ax1.set_title('Sector Accumulation Values')
    ax1.set_theta_zero_location('N')  # 0 degrees at the top
    ax1.set_theta_direction(-1)  # Clockwise
    
    # Add threshold circle
    threshold_line, = ax1.plot(np.linspace(0, 2*np.pi, 100), 
                             np.ones(100) * accumulator.threshold, 
                             'r--', alpha=0.7, label='Threshold')
    ax1.legend()
    
    # Initialize avoidance commands plot
    cmd_names = ['move_forward', 'move_backward', 'move_left', 'move_right', 
                'move_up', 'move_down', 'rotate_left', 'rotate_right']
    y_pos = np.arange(len(cmd_names))
    command_bars = ax2.barh(y_pos, np.zeros_like(y_pos), align='center', color='gray')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(cmd_names)
    ax2.set_xlim(0, 1.1)
    ax2.set_title('Avoidance Commands')
    
    # Current flow direction indicator
    flow_text = fig.text(0.5, 0.02, 'Flow Direction: None', ha='center')
    
    def update(frame):
        # Change flow direction every 10 frames
        if frame % 10 == 0:
            angle = (frame // 10) * np.pi/4  # Change direction every 45 degrees
            flow_direction = [np.cos(angle), np.sin(angle)]
            flow_text.set_text(f'Flow Direction: {angle:.1f} rad')
        else:
            angle = (frame // 10) * np.pi/4
            flow_direction = [np.cos(angle), np.sin(angle)]
        
        # Generate test data with this flow direction
        event_tensor = generate_test_event_data(flow_direction=flow_direction, magnitude=1.0)
        
        # Process through accumulator
        result = accumulator.process_event_frame(event_tensor)
        
        # Update polar plot
        for i, bar in enumerate(bars):
            bar.set_height(accumulator.accumulators[i])
            if accumulator.accumulators[i] > accumulator.threshold:
                bar.set_facecolor('red')
            else:
                bar.set_facecolor('blue')
        
        # Update command bars
        avoidance_commands = result['avoidance_commands']
        for i, bar in enumerate(command_bars):
            cmd_value = int(list(avoidance_commands.values())[i])
            bar.set_width(cmd_value)
            bar.set_color('green' if cmd_value else 'gray')
        
        # Return all artists that were updated
        return bars + command_bars + [flow_text, threshold_line]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=80, interval=200, blit=True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize sectorized obstacle avoidance')
    parser.add_argument('--mode', type=str, choices=['static', 'interactive'], 
                        default='interactive', help='Visualization mode')
    parser.add_argument('--sectors', type=int, default=8, 
                        help='Number of sectors for accumulation')
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        interactive_visualization()
        return
    
    # Static visualization mode
    accumulator = SectorizedSpikeAccumulator(
        num_sectors=args.sectors,
        threshold=2.0,
        decay_rate=0.8,
        device='cpu'  # Use CPU for visualization
    )
    
    try:
        while True:
            # Generate random flow direction for testing
            angle = np.random.uniform(0, 2*np.pi)
            flow_direction = [np.cos(angle), np.sin(angle)]
            print(f"Testing flow direction: [{flow_direction[0]:.2f}, {flow_direction[1]:.2f}]")
            
            # Generate test data
            event_tensor = generate_test_event_data(flow_direction=flow_direction)
            
            # Process the event tensor
            result = accumulator.process_event_frame(event_tensor)
            
            # Print the results
            print(f"Sector counts: {result['sector_counts']}")
            print(f"Avoidance commands: {result['avoidance_commands']}")
            
            # Visualize
            vis_img = create_sector_visualization(accumulator)
            cv2.imshow('Sector Visualization', vis_img)
            
            # Also show matplotlib visualization
            visualize_with_matplotlib(accumulator)
            
            key = cv2.waitKey(1000)
            if key == 27:  # ESC key
                break
                
    except KeyboardInterrupt:
        print("Exiting...")
    
    cv2.destroyAllWindows()
    plt.close('all')

if __name__ == "__main__":
    main() 