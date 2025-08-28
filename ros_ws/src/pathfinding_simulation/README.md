# Pathfinding Simulation Package

A ROS package implementing Multi-Heuristic A* (MHA*) pathfinding with Bayesian obstacle analysis and Building Information Modeling (BIM) integration for autonomous robot navigation in complex environments.

## Features

- **Multi-Heuristic A* Pathfinding**: Advanced pathfinding algorithm using multiple heuristics for optimal path planning
- **Bayesian Obstacle Analysis**: Dynamic obstacle danger assessment using OpenAI GPT integration
- **BIM Integration**: Building Information Modeling data integration for enhanced environment understanding
- **Repulsive Field Generation**: Dynamic repulsive potential fields for obstacle avoidance
- **Configuration Space Mapping**: Robot-aware obstacle inflation for safe navigation
- **Grid Visualization**: Real-time visualization of walkable/obstacle grids and path planning
- **Multiple Robot Support**: Compatible with TurtleBot3 and Unitree Go1 robots (simulation and real hardware)

## Dependencies

- ROS Noetic
- Python 3
- Open3D
- NumPy
- SciPy
- OpenAI API (optional, for Bayesian updates)
- Gazebo
- RViz
- TurtleBot3 packages (for TurtleBot3 simulation)
- Unitree packages (for Go1 simulation and real robot)

## Installation

1. **Clone the repository:**
```bash
cd ~/catkin_ws/src
git clone https://github.com/DiceLabs/bayesian-pathfinding.git pathfinding_simulation
```

2. **Install Python dependencies:**
```bash
pip3 install open3d numpy scipy requests
```

3. **Build the package:**
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Configuration

### OpenAI API Setup
For Bayesian obstacle analysis, set your OpenAI API key in the launch files:
```xml
<param name="api_key" value="your-openai-api-key-here"/>
```

### Robot Parameters
The package supports different robot configurations through ROS parameters:
- `robot_radius`: Robot's circular radius for configuration space (default: 0.14m for Go1, 0.105m for TurtleBot3)
- `safety_margin`: Additional safety margin around obstacles (default: 0.10m)
- `cell_size`: Grid resolution (default: 0.025m)
- `max_detect_z`: Maximum obstacle detection height (default: 1.5m)

## Usage

### TurtleBot3 Simulation

#### 1. Start the Gazebo simulation:
```bash
export TURTLEBOT3_MODEL=burger  # or waffle, waffle_pi
roslaunch pathfinding_simulation bim_world.launch
```

#### 2. Start the navigation stack:
```bash
roslaunch pathfinding_simulation navigation.launch
```

#### 3. Set navigation goals in RViz:
- Use "2D Pose Estimate" to set the robot's initial position
- Use "2D Nav Goal" to set navigation targets
- Watch the MHA* algorithm plan and execute paths

### Unitree Go1 Simulation

#### 1. Start the simulation in Gazebo:
```bash
roslaunch pathfinding_simulation robot_simulation.launch rname:=go1 rviz:=false
```

#### 2. Start the robot controller:
```bash
rosrun unitree_guide junior_ctrl
```
Press keys '2' and '5' to activate MoveBase mode.

#### 3. Start the navigation stack:
```bash
roslaunch pathfinding_simulation robot_navigation.launch rname:=go1
```

#### 4. Set navigation goals in RViz:
- Use "2D Pose Estimate" to set the robot's initial position
- Use "2D Nav Goal" to set navigation targets
- Add a MarkerArray display in RViz and set the topic to /grid_markers to visualize the occupancy grid and planned paths

### Real Unitree Go1 Robot

For deployment on the actual Unitree Go1 robot hardware:

#### Terminal 1 - Robot Interface:
```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
roslaunch unitree_legged_real real.launch ctrl_level:=highlevel
```

#### Terminal 2 - Pathfinding Walker:
```bash
source /opt/ros/noetic/setup.bash  
source ~/catkin_ws/devel/setup.bash
rosrun pathfinding_simulation pathfinding_walker.py
```

#### Terminal 3 - RViz Visualization:
```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
rosrun rviz rviz -d ~/catkin_ws/src/pathfinding_simulation/rviz/pathfinding.rviz
```

#### RViz Setup for Real Robot:
1. Add a **MarkerArray** display in RViz
2. Set the topic to `/grid_markers` to visualize the occupancy grid and planned paths
3. Use "2D Pose Estimate" to set the robot's initial position
4. Use "2D Nav Goal" to set navigation targets

**Note**: Ensure the real robot is properly calibrated and the workspace is clear of obstacles before running autonomous navigation.

## Real World Demo

Here is a picture from the real robot experiment:

![Real world demo](ros_ws/src/pathfinding_simulation/InShot_202ros_ws/media/realworld.jpg)

### Generating Occupancy Grids

To generate occupancy grid maps from STL mesh files:

```bash
cd ~/catkin_ws/src/pathfinding_simulation/src
PYTHONPATH=. python3 pathfinding_simulation/grid/make_occupancy.py \
  --input_mesh ../models/dicelab_bim/meshes/dicelab_BIM.stl \
  --output_pgm ../maps/dicelab_map.pgm \
  --output_yaml ../maps/dicelab_map.yaml \
  --robot_radius 0.14 \
  --safety_margin 0.10 \
  --cell_size 0.025
```

#### Parameters:
- `--input_mesh`: Path to input STL mesh file
- `--output_pgm`: Output PGM image file
- `--output_yaml`: Output YAML metadata file
- `--robot_radius`: Robot radius for configuration space
- `--safety_margin`: Additional safety margin
- `--cell_size`: Grid resolution in meters

## Troubleshooting

### Go1 Robot Control Issues

If you encounter the following error with `junior_ctrl`:
```
[ERROR] Function setProcessScheduler failed.
```

**Solution:**
1. Edit `/etc/security/limits.conf`:
```bash
sudo nano /etc/security/limits.conf
```

2. Add these lines (replace `<username>` with your username):
```
<username> hard rtprio 99
<username> soft rtprio 99
```

3. Save, close, and reboot the system.

### TF Frame Issues

If you experience TF transform errors:
1. Check that all required TF frames are being published
2. Verify frame names match your robot configuration
3. Use `rosrun tf view_frames` to visualize the TF tree
4. Use `rosrun tf tf_echo <source_frame> <target_frame>` to debug specific transforms

### Grid Generation Issues

If grid generation fails:
1. Verify the STL mesh file exists and is readable
2. Check that Open3D can load the mesh format
3. Ensure sufficient memory for large meshes
4. Verify output directory permissions

### Real Robot Connection Issues

For real Unitree Go1 deployment:
1. Ensure proper network connection between control computer and robot
2. Verify the robot's IP address configuration
3. Check that the `unitree_legged_real` package is properly installed
4. Confirm the robot is in the correct control mode before starting navigation

## Package Structure

```
pathfinding_simulation/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── README.md
├── config/
│   └── diff_drive_controller.yaml  # Robot controller configuration
├── launch/
│   ├── amcl.launch                 # AMCL localization
│   ├── bim_world.launch           # TurtleBot3 simulation launch
│   ├── navigation.launch          # TurtleBot3 navigation
│   ├── robot_navigation.launch    # Go1 navigation
│   └── robot_simulation.launch    # Go1 simulation launch
├── maps/
│   ├── dicelab_map.pgm            # Generated occupancy grid map
│   └── dicelab_map.yaml           # Map metadata
├── models/
│   └── dicelab_bim/               # BIM models and data
│       ├── meshes/
│       │   ├── dicelab_BIM.stl    # Main BIM mesh file
│       │   ├── dicelab_BIM.dae    # Collada format
│       │   ├── dicelab_BIM.fbx    # FBX format
│       │   ├── extracted_objects.json           # BIM object data
│       │   └── extracted_objects_with_danger.json # Bayesian danger data
│       ├── model.config           # Gazebo model configuration
│       └── model.sdf              # SDF model definition
├── rviz/
│   └── pathfinding.rviz          # RViz configuration for pathfinding
├── scripts/
│   ├── pathfinding_node          # Main pathfinding ROS node (executable)
│   ├── path_follower_node        # Path following controller (executable)
│   └── smha_star_server          # SMHA* server (executable)
├── src/pathfinding_simulation/
│   ├── __init__.py
│   ├── pathfinding_node.py       # Main pathfinding ROS node
│   ├── path_follower_node.py     # Path following controller
│   ├── pathfinding_walker.py     # Real robot pathfinding walker
│   ├── pathfinding/
│   │   ├── __init__.py
│   │   └── mhastar.py            # Multi-Heuristic A* implementation
│   ├── grid/
│   │   ├── __init__.py
│   │   ├── grid_generator_open3d.py # Grid generation from STL meshes
│   │   ├── node.py               # Grid node representation
│   │   └── make_occupancy.py     # Standalone occupancy grid generator
│   ├── fields/
│   │   ├── __init__.py
│   │   ├── repulsive_field.py    # Repulsive potential field computation
│   │   └── dynamic_repulsion_manager.py # Dynamic obstacle management
│   └── utils/
│       ├── __init__.py
│       ├── bayesian_repulsion_updater.py # OpenAI-based Bayesian analysis
│       ├── bim_integration.py    # BIM data integration
│       └── extract_fbx_objects.py # FBX object extraction utility
├── srv/
│   └── StartGoalPath.srv         # Path planning service definition
└── worlds/
    └── dicelab.world             # Gazebo world file
```

## Algorithm Details

### Multi-Heuristic A* (MHA*)
The pathfinding algorithm uses two heuristics:
- **h₀**: Euclidean distance to goal (admissible)
- **h₁**: Repulsive potential field values (inadmissible)

Parameters:
- `w1`: Weight for heuristic scaling (default: 1.0)
- `w2`: Weight for inadmissible heuristic (default: 2.0)
- `gamma`: Repulsive potential scaling factor (default: 5.0)

### Bayesian Obstacle Analysis
The system uses OpenAI's GPT to assess obstacle danger based on:
- Object names from BIM data
- Environmental context
- Dynamic activity levels

### Repulsive Field Computation
Obstacles generate repulsive potential fields using:
- Exponential decay: `strength = gain × exp(-distance)`
- Configurable influence radius
- Bayesian-updated gain values
