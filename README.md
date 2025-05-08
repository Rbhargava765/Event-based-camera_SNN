# Event-based SNN Optical Flow for Obstacle Avoidance

This project implements an ultra-low latency obstacle avoidance system for drones using event-based cameras, Spiking Neural Networks (SNN), and optical flow. The system follows a "reflex-based" approach where detected motion in specific sectors of the visual field triggers immediate avoidance manoeuvres.

## Architecture

The system consists of three main components:

1. **Event-based SNN Optical Flow Estimator**: Based on the work from [Frontiers in Neuroscience](https://doi.org/10.3389/fnins.2023.1160034), this component processes event camera data to estimate optical flow fields.

2. **Sectorized Spike Accumulation**: Divides the visual field into sectors and accumulates optical flow magnitude in each sector using leaky integration.

3. **Spiking Reflex Module**: Generates avoidance commands when accumulated flow in any sector exceeds a threshold.

The complete pipeline has minimal computational requirements and can run efficiently on embedded hardware, making it ideal for resource-constrained platforms like drones.

## Key Features

- **Ultra-low latency**: The event-based approach allows for microsecond-level response times
- **Minimal compute**: Simple sector-based accumulation and threshold operations
- **Deterministic behaviour**: Fixed thresholds and event-driven logic ensure predictable timing
- **PX4/ROS integration**: Ready for integration with standard drone control stacks
- **Scalable design**: Easily add more sectors or adjust thresholds

## Getting Started

### Prerequisites

- Python 3.6+
- PyTorch 1.7+
- SpikingJelly 0.0.0.0.12
- OpenCV
- NumPy
- Matplotlib (for visualization)

For ROS integration:
- ROS Noetic/Melodic
- MAVROS package

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/event-based-snn-obstacle-avoidance.git
cd event-based-snn-obstacle-avoidance
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained SNN model:
```bash
mkdir -p OF_EV_SNN-main/examples
# Download the pre-trained model to OF_EV_SNN-main/examples/checkpoint_epoch34.pth
```

### Basic Usage

To run the obstacle avoidance controller in standalone mode:

```bash
python obstacle_avoidance_controller.py
```

To visualize the sectorized accumulation and avoidance commands:

```bash
python visualize_sectors.py --mode interactive
```

### ROS Integration

To run the controller with ROS and connect to a PX4 flight controller:

```bash
rosrun event_based_snn_obstacle_avoidance ros_integration.py
```

## System Architecture Details

### 1. Event Camera Processing

The system uses an event camera (also known as a Dynamic Vision Sensor or DVS) that outputs events only when pixel brightness changes. These events are accumulated into event frames and processed through the SNN to estimate optical flow.

### 2. SNN-based Optical Flow Estimation

The optical flow estimation is performed by a spiking neural network with the following architecture:

- Encoder: 3D convolutional layers that process temporal sequences of event frames
- Angular loss function: Improves accuracy and generalization
- Decoder: Generates dense optical flow fields

### 3. Sectorized Spike Accumulation

The visual field is divided into sectors (typically 8), and the optical flow magnitude in each sector is accumulated using a leaky integrator:

```
A(t) = decay_rate * A(t-1) + sum(flow_magnitude(sector))
```

### 4. Reflex-based Avoidance

When the accumulated value in any sector exceeds a predefined threshold, an avoidance manoeuvre is triggered:

- If high flow on the right → turn left
- If high flow on the left → turn right
- If high flow on top → move down
- If high flow on bottom → move up

### 5. Controller Integration

The system integrates with flight controllers using either:

- **Velocity commands**: Published to `/mavros/setpoint_velocity/cmd_vel`
- **RC override**: Direct control via `/mavros/rc/override`

## Customization

You can customize the system by adjusting these parameters:

- `num_sectors`: Number of sectors to divide the visual field into
- `threshold`: Threshold for triggering avoidance
- `decay_rate`: Decay rate for the leaky integrator

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original SNN Optical Flow code from [Optical Flow estimation from Event Cameras and Spiking Neural Networks](https://github.com/neuromorphic-paris/OF_EV_SNN)
- Event camera processing based on [SpikingJelly](https://github.com/fangwei123456/spikingjelly) 