#!/usr/bin/env python3
"""
ROS integration for the SNN-based Optical Flow Obstacle Avoidance Controller

This script shows how to connect the SectorizedSpikeAccumulator to ROS and the PX4 flight stack.
It subscribes to event camera data, processes it through the SNN to estimate optical flow,
and then uses the sectorized spike accumulation to generate avoidance commands.

Requirements:
- ROS Noetic/Melodic
- PX4 MAVROS package
- PyTorch with CUDA support
- SpikingJelly 0.0.0.0.12 or compatible version
"""

import rospy
import numpy as np
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import OverrideRCIn

from obstacle_avoidance_controller import SectorizedSpikeAccumulator

class ROSSpikeObstacleAvoidance:
    """ROS node for event-based SNN obstacle avoidance"""
    
    def __init__(self):
        """Initialize the ROS node and related components"""
        # Initialize ROS node
        rospy.init_node('snn_obstacle_avoidance', anonymous=True)
        
        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Initialize the SNN-based obstacle avoidance controller
        self.accumulator = SectorizedSpikeAccumulator(
            num_sectors=8,
            threshold=2.0,
            decay_rate=0.9,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Create publishers and subscribers
        self.setup_ros_interfaces()
        
        # Safety parameters
        self.last_event_time = rospy.Time.now()
        self.timeout_duration = rospy.Duration(0.5)  # 500ms timeout
        
        rospy.loginfo("SNN Obstacle Avoidance node initialized")
    
    def setup_ros_interfaces(self):
        """Setup ROS publishers and subscribers"""
        # Subscribe to event camera data
        # Note: The exact topic and message type will depend on your event camera driver
        rospy.Subscriber("/event_camera/events", Image, self.event_callback, queue_size=1)
        
        # For standard frame-based event representations (e.g., event frame, time surfaces)
        rospy.Subscriber("/event_camera/event_frame", Image, self.event_frame_callback, queue_size=1)
        
        # Publishers for control commands
        # For direct velocity control
        self.vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)
        
        # For RC override (direct channel control)
        self.rc_override_pub = rospy.Publisher("/mavros/rc/override", OverrideRCIn, queue_size=10)
        
        # For visualization
        self.viz_pub = rospy.Publisher("/obstacle_avoidance/visualization", Image, queue_size=1)
    
    def event_callback(self, msg):
        """
        Callback for raw event data (depends on the specific event camera driver)
        """
        rospy.logwarn_once("Raw event callback not implemented. Using event frames instead.")
        
    def event_frame_callback(self, msg):
        """
        Callback for event frame data (accumulated events)
        
        Args:
            msg: ROS Image message containing event frame data
        """
        try:
            # Update the last event time
            self.last_event_time = rospy.Time.now()
            
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            
            # Process the event frame for optical flow
            event_tensor = self.convert_to_event_tensor(cv_image)
            
            # Get avoidance commands from the accumulator
            result = self.accumulator.process_event_frame(event_tensor)
            
            # Send commands to the flight controller
            self.send_avoidance_commands(result['avoidance_commands'])
            
            # Publish visualization if there are subscribers
            if self.viz_pub.get_num_connections() > 0:
                self.publish_visualization()
                
        except Exception as e:
            rospy.logerr(f"Error processing event frame: {e}")
    
    def convert_to_event_tensor(self, cv_image):
        """
        Convert OpenCV image to event tensor format required by the SNN model
        
        Args:
            cv_image: OpenCV image containing event data
            
        Returns:
            torch.Tensor: Event tensor in format [1, 2, time_steps, height, width]
        """
        height, width = cv_image.shape
        
        # Create a 21-frame sequence (as required by the SNN model)
        event_tensor = torch.zeros(1, 2, 21, height, width, dtype=torch.float32)
        
        normalized_frame = cv_image.astype(np.float32) / 255.0
        
        for t in range(21):
            event_tensor[0, 0, t] = torch.from_numpy(normalized_frame)
            event_tensor[0, 1, t] = torch.from_numpy(normalized_frame)
        
        return event_tensor
    
    def send_avoidance_commands(self, commands):
        """
        Send avoidance commands to the flight controller
        
        Args:
            commands (dict): Dictionary of avoidance commands
        """
        # Method 1: Using velocity commands
        self.send_velocity_commands(commands)
        
        # Method 2: Using RC override (direct channel control)
        # self.send_rc_override(commands)
    
    def send_velocity_commands(self, commands):
        """
        Send velocity commands to the flight controller
        
        Args:
            commands (dict): Dictionary of avoidance commands
        """
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        
        # Set default velocity (forward)
        linear_speed = 0.5  # m/s
        angular_speed = 0.3  # rad/s
        
        # Set linear velocities
        if commands['move_forward']:
            msg.twist.linear.x = linear_speed
        elif commands['move_backward']:
            msg.twist.linear.x = -linear_speed
        else:
            msg.twist.linear.x = 0.0
            
        if commands['move_right']:
            msg.twist.linear.y = -linear_speed
        elif commands['move_left']:
            msg.twist.linear.y = linear_speed
        else:
            msg.twist.linear.y = 0.0
            
        if commands['move_up']:
            msg.twist.linear.z = linear_speed
        elif commands['move_down']:
            msg.twist.linear.z = -linear_speed
        else:
            msg.twist.linear.z = 0.0
        
        # Set angular velocities (yaw control)
        if commands['rotate_right']:
            msg.twist.angular.z = -angular_speed
        elif commands['rotate_left']:
            msg.twist.angular.z = angular_speed
        else:
            msg.twist.angular.z = 0.0
        
        # Publish the message
        self.vel_pub.publish(msg)
    
    def send_rc_override(self, commands):
        """
        Send RC override commands to the flight controller
        
        Args:
            commands (dict): Dictionary of avoidance commands
        """
        # RC override message
        msg = OverrideRCIn()
        
        # Default values (mid-point for most channels)
        # PX4 typically uses 1500 as the center value
        default_value = 1500
        
        # Initialize all channels to default
        for i in range(8):
            msg.channels[i] = default_value
        
        # Define channel mappings (these will depend on your RC configuration)
        # Standard mapping:
        # Channel 1: Roll (left/right)
        # Channel 2: Pitch (forward/backward)
        # Channel 3: Throttle (up/down)
        # Channel 4: Yaw (rotate left/right)
        
        # Set channel values based on commands
        if commands['move_forward']:
            msg.channels[1] = default_value + 200  # Pitch forward
        elif commands['move_backward']:
            msg.channels[1] = default_value - 200  # Pitch backward
            
        if commands['move_right']:
            msg.channels[0] = default_value + 200  # Roll right
        elif commands['move_left']:
            msg.channels[0] = default_value - 200  # Roll left
            
        if commands['move_up']:
            msg.channels[2] = default_value + 200  # Increase throttle
        elif commands['move_down']:
            msg.channels[2] = default_value - 200  # Decrease throttle
            
        if commands['rotate_right']:
            msg.channels[3] = default_value + 200  # Yaw right
        elif commands['rotate_left']:
            msg.channels[3] = default_value - 200  # Yaw left
        
        # Publish the message
        self.rc_override_pub.publish(msg)
    
    def publish_visualization(self):
        """Publish visualization of the current obstacle avoidance state"""
        try:
            # Import here to avoid dependency for those who don't need visualization
            from visualize_sectors import create_sector_visualization
            import cv2
            
            # Create visualization image
            vis_img = create_sector_visualization(self.accumulator)
            
            # Convert to ROS Image message
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
            vis_msg.header.stamp = rospy.Time.now()
            
            # Publish
            self.viz_pub.publish(vis_msg)
            
        except Exception as e:
            rospy.logerr(f"Error publishing visualization: {e}")
    
    def check_timeout(self):
        """
        Check if we've timed out waiting for event data
        Returns True if timed out, False otherwise
        """
        current_time = rospy.Time.now()
        elapsed = current_time - self.last_event_time
        
        return elapsed > self.timeout_duration
    
    def run(self):
        """Main run loop"""
        rate = rospy.Rate(30)  # 30Hz
        
        while not rospy.is_shutdown():
            # Check for timeout (no recent event data)
            if self.check_timeout():
                rospy.logwarn_throttle(5.0, "No recent event data received")
                
            rate.sleep()


if __name__ == "__main__":
    try:
        node = ROSSpikeObstacleAvoidance()
        node.run()
    except rospy.ROSInterruptException:
        pass 