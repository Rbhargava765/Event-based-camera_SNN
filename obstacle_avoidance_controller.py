import torch
import numpy as np
import time
from spikingjelly.clock_driven import functional
import sys
import os

# Add the project directory to the path so we can import the network
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from network_3d.poolingNet_cat_1res import NeuronPool_Separable_Pool3d

class SectorizedSpikeAccumulator:
    """
    Implements the Sectorized Spike Accumulation for obstacle avoidance based on optical flow
    from the event camera SNN.
    """
    def __init__(self, num_sectors=8, threshold=0.5, decay_rate=0.8, device='cuda'):
        """
        Initialize the sectorized spike accumulator.
        
        Args:
            num_sectors (int): Number of sectors to divide the visual field into
            threshold (float): Threshold for triggering avoidance reflex
            decay_rate (float): Decay rate for the leaky integrator
            device (str): Device to run computations on ('cuda' or 'cpu')
        """
        self.num_sectors = num_sectors
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize accumulators for each sector
        self.accumulators = np.zeros(num_sectors)
        
        # Initialize SNN model for optical flow estimation
        self.model = self._load_model()
        
        # Define sector masks (dividing the image into angular sectors)
        self.sector_masks = self._create_sector_masks((480, 640))
    
    def _load_model(self):
        """Load the pre-trained SNN model for optical flow estimation"""
        model = NeuronPool_Separable_Pool3d(multiply_factor=35.0).to(self.device)
        
        # Load the pre-trained weights if available
        model_path = 'OF_EV_SNN-main/examples/checkpoint_epoch34.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Could not load model from {model_path}")
        
        model.eval()
        return model
    
    def _create_sector_masks(self, image_shape):
        """
        Create masks for each sector in the image.
        
        Args:
            image_shape (tuple): Shape of the image (height, width)
        
        Returns:
            list: List of binary masks for each sector
        """
        height, width = image_shape
        center_y, center_x = height // 2, width // 2
        
        # Create meshgrid for calculating angles
        y, x = np.mgrid[:height, :width]
        y = -(y - center_y)  # Flip y axis to match conventional coordinates
        x = x - center_x
        
        # Calculate angles for each pixel (in radians)
        angles = np.arctan2(y, x)
        # Convert to degrees and shift range from [-180, 180] to [0, 360]
        angles_deg = (np.degrees(angles) + 360) % 360
        
        # Create masks for each sector
        sector_size = 360 / self.num_sectors
        masks = []
        
        for i in range(self.num_sectors):
            sector_start = i * sector_size
            sector_end = (i + 1) * sector_size
            mask = np.logical_and(angles_deg >= sector_start, angles_deg < sector_end)
            masks.append(mask)
        
        return masks
    
    def process_event_frame(self, event_tensor):
        """
        Process an event frame through the SNN and update the accumulators.
        
        Args:
            event_tensor (torch.Tensor): Event tensor of shape [1, 2, time_steps, height, width]
        
        Returns:
            dict: Spike counts per sector and avoidance commands
        """
        # Process through SNN model
        with torch.no_grad():
            functional.reset_net(self.model)
            event_tensor = event_tensor.to(self.device)
            _, _, _, pred = self.model(event_tensor)
        
        # Extract optical flow vectors (x and y components)
        flow_x = pred[0, 0].cpu().numpy()
        flow_y = pred[0, 1].cpu().numpy()
        
        # Calculate flow magnitude
        flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)
        
        # Update accumulators for each sector
        sector_counts = np.zeros(self.num_sectors)
        for i, mask in enumerate(self.sector_masks):
            # Calculate the sum of flow magnitude in this sector
            sector_flow = flow_magnitude * mask
            
            # Update the accumulator with leaky integration
            self.accumulators[i] = self.decay_rate * self.accumulators[i] + np.sum(sector_flow)
            sector_counts[i] = self.accumulators[i]
        
        # Generate avoidance commands based on accumulator values
        avoidance_commands = self._generate_avoidance_commands()
        
        return {
            'sector_counts': sector_counts,
            'avoidance_commands': avoidance_commands
        }
    
    def _generate_avoidance_commands(self):
        """
        Generate avoidance commands based on accumulator values.
        
        Returns:
            dict: Dictionary of avoidance commands
        """
        # Check which sectors exceed the threshold
        triggered_sectors = np.where(self.accumulators > self.threshold)[0]
        
        # If no sectors are triggered, continue forward
        if len(triggered_sectors) == 0:
            return {
                'move_forward': True,
                'move_backward': False,
                'move_left': False,
                'move_right': False,
                'move_up': False,
                'move_down': False,
                'rotate_left': False,
                'rotate_right': False
            }
        
        # Find the sector with the highest accumulator value
        max_sector = np.argmax(self.accumulators)
        
        # Define commands based on sector
        # This mapping depends on how you've defined your sectors
        # Assuming sector 0 is straight ahead, and they proceed clockwise
        commands = {
            'move_forward': False,
            'move_backward': False,
            'move_left': False,
            'move_right': False,
            'move_up': False,
            'move_down': False,
            'rotate_left': False,
            'rotate_right': False
        }
        
        # Right side sectors (0, 1, 7)
        if max_sector in [0, 1, 7]:
            commands['rotate_left'] = True
        
        # Left side sectors (3, 4, 5)
        elif max_sector in [3, 4, 5]:
            commands['rotate_right'] = True
        
        # Top sectors (1, 2, 3)
        elif max_sector in [1, 2, 3]:
            commands['move_down'] = True
        
        # Bottom sectors (5, 6, 7)
        elif max_sector in [5, 6, 7]:
            commands['move_up'] = True
        
        return commands
    
    def reset_accumulators(self):
        """Reset all accumulators to zero"""
        self.accumulators = np.zeros(self.num_sectors)


# Function to connect with ROS (placeholder)
def connect_to_ros():
    """
    Connect to ROS and set up publishers/subscribers.
    This is a placeholder - implement actual ROS connection based on your setup.
    """
    print("Connecting to ROS...")
    # Here you would import rospy and create publishers/subscribers
    return None


# Function to connect with PX4 (placeholder)
def connect_to_px4():
    """
    Connect to PX4 flight controller.
    This is a placeholder - implement actual PX4 connection based on your setup.
    """
    print("Connecting to PX4...")
    # Here you would import appropriate PX4 libraries
    return None


def main():
    """Main function to run the obstacle avoidance controller"""
    # Initialize the sectorized spike accumulator
    accumulator = SectorizedSpikeAccumulator(
        num_sectors=8,
        threshold=0.5,
        decay_rate=0.8,
        device='cuda'
    )
    
    # Connect to ROS and PX4 (implement these functions based on your setup)
    ros_node = connect_to_ros()
    px4_connection = connect_to_px4()
    
    print("Obstacle avoidance controller initialized successfully!")
    print("Press Ctrl+C to exit.")
    
    try:
        while True:
            # This is where you would read event camera data from ROS
            # For testing, we'll create a dummy event tensor
            event_tensor = torch.randn(1, 2, 11, 480, 640)
            
            # Process the event tensor
            result = accumulator.process_event_frame(event_tensor)
            
            # Print the results
            print(f"Sector counts: {result['sector_counts']}")
            print(f"Avoidance commands: {result['avoidance_commands']}")
            
            # Here you would send commands to PX4
            # For example: send_commands_to_px4(px4_connection, result['avoidance_commands'])
            
            time.sleep(0.1)  # 10 Hz update rate
    
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main() 