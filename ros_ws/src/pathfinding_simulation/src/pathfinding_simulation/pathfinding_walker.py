#!/usr/bin/env python3


import rospy
import numpy as np
import math
import os
import requests
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from unitree_legged_msgs.msg import HighCmd, HighState
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA

# Import pathfinding modules 
try:
    import sys
    import os
    
   
    pathfinding_path = '/home/dicelabs/catkin_ws/src/pathfinding_simulation/src'
    if pathfinding_path not in sys.path:
        sys.path.insert(0, pathfinding_path)
    
    
    from pathfinding_simulation.grid.grid_generator_open3d import GridGenerator
    from pathfinding_simulation.pathfinding.mhastar import MultiHeuristicAStar  
    from pathfinding_simulation.fields.repulsive_field import compute_repulsive_field
    from pathfinding_simulation.utils.bim_integration import create_complete_integration
    from pathfinding_simulation.utils.bayesian_repulsion_updater import BayesianRepulsionUpdater
    
    HAS_PATHFINDING = True
    print("SUCCESS: MHA* pathfinding modules loaded!")
    
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print(f"Checking if files exist...")
    
    # Debug the file paths
    base_path = '/home/dicelabs/catkin_ws/src/pathfinding_simulation/src/pathfinding_simulation'
    files_to_check = [
        'grid/grid_generator_open3d.py',
        'pathfinding/mhastar.py', 
        'fields/repulsive_field.py',
        'utils/bim_integration.py',
        'utils/bayesian_repulsion_updater.py'
    ]
    
    for file_path in files_to_check:
        full_path = os.path.join(base_path, file_path)
        exists = os.path.exists(full_path)
        print(f"  {file_path}: {'EXISTS' if exists else 'MISSING'}")
    
    HAS_PATHFINDING = False

class PathfindingWalker:
    def __init__(self):
        rospy.init_node('pathfinding_walker')
        
        # Publishers
        self.cmd_pub = rospy.Publisher('/high_cmd', HighCmd, queue_size=1000)
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=1)
        self.grid_marker_pub = rospy.Publisher('/grid_markers', MarkerArray, queue_size=1, latch=True)
        
        # Subscribers
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.start_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        rospy.Subscriber('/high_state', HighState, self.state_callback)
        # Navigation state - initialize before pathfinding setup
        self.current_path = []
        self.goal_idx = 0
        self.robot_pose = [-0.385, 1.4, 0.0]  # x, y, yaw - matches robot spawn position
        self.start_pos = [-0.385, 1.4, 0.0]  # Hardcoded start position
        self.api_key = "API_KEY" 
        self.goal_pos = None
        self.path_following = False
        
        # Pathfinding setup
        self.setup_pathfinding()
        
        # Control parameters
        self.linear_speed = rospy.get_param('~linear_speed', 0.3)
        self.angular_speed = rospy.get_param('~angular_speed', 0.5)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.15)
        self.angle_tolerance = rospy.get_param('~angle_tolerance', 0.1)
        
        # Motion control
        self.motiontime = 0
        self.rate = rospy.Rate(500) 
        
        rospy.loginfo("Pathfinding Walker initialized. Use RViz to set start and goal.")
        
    def setup_pathfinding(self):
        """Setup pathfinding system"""
        if not HAS_PATHFINDING:
            rospy.logwarn("Pathfinding not available")
            self.grid_gen = None
            return
            
        try:
            # Get parameters for grid generation
            default_mesh = '/home/dicelabs/catkin_ws/src/pathfinding_simulation/models/dicelab_bim/meshes/dicelab_BIM.stl'
            mesh_file = rospy.get_param('~mesh_file', default_mesh)
            
            if not os.path.exists(mesh_file):
                rospy.logerr(f"Mesh file not found: {mesh_file}")
                self.grid_gen = None
                return
                
            # Initialize grid generator
            self.grid_gen = GridGenerator(
                mesh_file=mesh_file,
                cell_size=rospy.get_param('~cell_size', 0.025),
                robot_radius=rospy.get_param('~robot_radius', 0.14),
                margin=rospy.get_param('~safety_margin', 0.01)
            )
            
            # Generate grid
            walkable_points, obstacle_points = self.grid_gen.generate_grid()
            rospy.loginfo(f"Grid generated: {len(walkable_points)} walkable, {len(obstacle_points)} obstacles")
            
            # Setup BIM integration
            default_bim = '/home/dicelabs/catkin_ws/src/pathfinding_simulation/models/dicelab_bim/meshes/extracted_objects.json'
            bim_file = rospy.get_param('~bim_data', default_bim)
            
            if os.path.exists(bim_file):
                self.bim_integration = create_complete_integration(bim_file)
                
                # Setup Bayesian updater if API key provided
                if self.api_key.strip():
                    try:
                        rospy.loginfo(f"[DEBUG] Initializing Bayesian updater with API key: {self.api_key[:10]}...")
                        self.bayesian_updater = BayesianRepulsionUpdater(
                            api_key=self.api_key,
                            obstacles=self.bim_integration.get_obstacles_for_bayesian()
                        )
                        rospy.loginfo("[DEBUG] Bayesian updater initialized successfully")
                    except Exception as e:
                        rospy.logerr(f"[DEBUG] Failed to initialize Bayesian updater: {e}")
                        self.bayesian_updater = None
                else:
                    rospy.logwarn("[DEBUG] No API key provided - skipping Bayesian updates")
                    self.bayesian_updater = None
                
                # Compute repulsive field
                repulsive_obstacles = self.bim_integration.get_obstacles_for_repulsive_field()
                compute_repulsive_field(
                    grid=self.grid_gen.get_grid(),
                    obstacles=repulsive_obstacles
                )
                rospy.loginfo("Repulsive field computed")
                
                # Publish grid markers for RViz
                self.publish_grid_markers()
            else:
                self.bim_integration = None
                self.bayesian_updater = None
                
        except Exception as e:
            rospy.logerr(f"Error setting up pathfinding: {e}")
            self.grid_gen = None
    
    def publish_grid_markers(self):
        """Publish grid markers for RViz visualization"""
        if not self.grid_gen:
            return
            
        rospy.loginfo("Publishing grid markers to RViz...")
        marker_array = MarkerArray()
        marker_id = 0
        
        grid = self.grid_gen.get_grid()
        for x in range(self.grid_gen.grid_size_x):
            for y in range(self.grid_gen.grid_size_y):
                node = grid[x][y]
                if node is None:
                    continue
                    
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "grid"
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = node.world_position[0]
                marker.pose.position.y = node.world_position[1]
                marker.pose.position.z = 0.01
                marker.scale.x = self.grid_gen.node_diameter
                marker.scale.y = self.grid_gen.node_diameter
                marker.scale.z = 0.01
                
                # Color based on walkability and repulsive potential
                if node.walkable:
                    intensity = min(1.0, node.repulsive_potential * 0.5)
                    marker.color = ColorRGBA(r=intensity, g=1.0-intensity, b=0.0, a=0.3)
                else:
                    marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)
                    
                marker.lifetime = rospy.Duration(0)
                marker_array.markers.append(marker)
                marker_id += 1
        
        rospy.loginfo(f"Total markers published: {marker_id}")
        self.grid_marker_pub.publish(marker_array)
    
    def publish_grid_markers(self):
        """Publish grid markers for RViz visualization with path overlay"""
        if not self.grid_gen:
            return
            
        rospy.loginfo("Publishing grid markers to RViz...")
        marker_array = MarkerArray()
        marker_id = 0
        
        # Get current path positions for highlighting
        current_path_positions = []
        if self.current_path:
            current_path_positions = [[waypoint[0], waypoint[1], waypoint[2]] for waypoint in self.current_path]
        
        path_positions = set()
        for pos in current_path_positions:
            path_positions.add((round(pos[0], 3), round(pos[1], 3)))
        
        grid = self.grid_gen.get_grid()
        for x in range(self.grid_gen.grid_size_x):
            for y in range(self.grid_gen.grid_size_y):
                node = grid[x][y]
                if node is None:
                    continue
                    
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "grid"
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = node.world_position[0]
                marker.pose.position.y = node.world_position[1]
                marker.pose.position.z = 0.01
                marker.scale.x = self.grid_gen.node_diameter
                marker.scale.y = self.grid_gen.node_diameter
                marker.scale.z = 0.02
                
                # Check if this cell is on the path
                cell_pos = (round(node.world_position[0], 3), round(node.world_position[1], 3))
                if cell_pos in path_positions:
                    # Highlight path cells in blue 
                    marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
                    marker.scale.z = 0.05
                elif node.walkable:
                    intensity = min(1.0, node.repulsive_potential * 0.5)
                    marker.color = ColorRGBA(r=intensity, g=1.0-intensity, b=0.0, a=0.3)
                else:
                    marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)
                    
                marker.lifetime = rospy.Duration(0)
                marker_array.markers.append(marker)
                marker_id += 1

        # Add waypoint spheres
        for pos in current_path_positions:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = "path_waypoints"
            m.id = marker_id
            marker_id += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = pos[0]
            m.pose.position.y = pos[1]
            m.pose.position.z = 0.1
            m.scale.x = m.scale.y = m.scale.z = 0.08
            m.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.9)
            m.lifetime = rospy.Duration(0)
            marker_array.markers.append(m)
        
        rospy.loginfo(f"Total markers published: {marker_id}")
        self.grid_marker_pub.publish(marker_array)
    
    def start_following_callback(self, msg):
        """Start path following when commanded via topic"""
        if hasattr(self, 'waiting_for_confirmation') and self.waiting_for_confirmation:
            self.waiting_for_confirmation = False
            self.path_following = True
            rospy.loginfo("=" * 50)
            rospy.loginfo("STARTING PATH FOLLOWING!")
            rospy.loginfo("Robot will now follow the blue path...")
            rospy.loginfo("=" * 50)
    
    def start_callback(self, msg):
        """Skip RViz start input - using hardcoded start position"""
        rospy.loginfo("Using hardcoded start position - ignoring RViz start input")
        # self.start_pos is already set in __init__
        self.try_plan_path()
    
    def goal_callback(self, msg):
        """Set goal position from RViz"""
        pos = msg.pose.position
        self.goal_pos = [pos.x, pos.y, 0.0]
        rospy.loginfo(f"Goal position set: {self.goal_pos}")
        self.try_plan_path()
    
    def state_callback(self, msg):
        """Update robot state"""
        # Simple pose estimation from robot state
     
        pass
    
    def try_plan_path(self):
        """Plan path when both start and goal are set"""
        if not self.start_pos or not self.goal_pos:
            return
            
        if not self.grid_gen:
            # Fallback: direct path
            self.current_path = [self.start_pos, self.goal_pos]
            rospy.loginfo("Using direct path (no pathfinding available)")
        else:
            try:
                # Perform Bayesian update if available
                if self.bayesian_updater:
                    try:
                        rospy.loginfo("[DEBUG] Performing Bayesian update...")
                        self.bayesian_updater.perform_update()
                        rospy.loginfo("[DEBUG] Bayesian update completed successfully")
                        
                        # Update repulsive field with new gains
                        if self.bim_integration:
                            rospy.loginfo("[DEBUG] Updating repulsive gains...")
                            self.bim_integration.update_repulsive_gains()
                            repulsive_obstacles = self.bim_integration.get_obstacles_for_repulsive_field()
                            compute_repulsive_field(
                                grid=self.grid_gen.get_grid(),
                                obstacles=repulsive_obstacles
                            )
                            rospy.loginfo("[DEBUG] Repulsive field updated with Bayesian gains")
                            
                    except requests.exceptions.RequestException as e:
                        rospy.logwarn(f"[DEBUG] Network error during Bayesian update: {e}")
                        rospy.logwarn("[DEBUG] Continuing with existing repulsive field values...")
                    except Exception as e:
                        rospy.logwarn(f"[DEBUG] Bayesian update failed: {e}")
                        rospy.logwarn("[DEBUG] Continuing with existing repulsive field values...")
                else:
                    rospy.loginfo("[DEBUG] No Bayesian updater available - using static repulsive field")
                
                # Initialize MHA* pathfinder
                mha_star = MultiHeuristicAStar(
                    grid_generator=self.grid_gen,
                    start_pos=self.start_pos,
                    goal_pos=self.goal_pos
                )
                
                # Plan path
                obstacles = None
                if self.bim_integration:
                    obstacles = self.bim_integration.get_obstacles_for_repulsive_field()
                
                path_nodes = []
                def capture_path(nodes):
                    nonlocal path_nodes
                    path_nodes = nodes
                    
                mha_star.plan_path(
                    start_position=self.start_pos,
                    goal_position=self.goal_pos,
                    bayes_manager=self.bayesian_updater,
                    obstacles=obstacles,
                    visualize_fn=capture_path
                )
                
                if path_nodes:
                    self.current_path = [[node.world_position[0], node.world_position[1], 0.0] for node in path_nodes]
                    rospy.loginfo(f"Path planned with {len(self.current_path)} waypoints")
                else:
                    rospy.logwarn("No path found")
                    return
                    
            except Exception as e:
                rospy.logerr(f"Path planning failed: {e}")
                return
        
        # Start path following
        self.goal_idx = 0
        self.path_following = False  # Don't start immediately
        self.robot_pose = list(self.start_pos)  # Initialize robot pose
        self.waiting_for_confirmation = True
        
        # Publish path for visualization
        self.publish_path()
        
        # Update grid markers to show the path
        if self.grid_gen:
            self.publish_grid_markers()
        
  
        rospy.loginfo("PATH PLANNED AND VISUALIZED!")
        rospy.loginfo("Check RViz to see the blue path.")
        rospy.loginfo("To START path following, run in another terminal:")
        rospy.loginfo("rostopic pub -1 /start_following std_msgs/Empty")

        
        # Set up subscriber for start command
        if not hasattr(self, 'start_sub'):
            from std_msgs.msg import Empty
            self.start_sub = rospy.Subscriber('/start_following', Empty, self.start_following_callback)
    
    def publish_path(self):
        """Publish path for RViz visualization"""
        if not self.current_path:
            return
            
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()
        
        for waypoint in self.current_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = waypoint[0]
            pose.pose.position.y = waypoint[1] 
            pose.pose.position.z = waypoint[2]
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
    
    @staticmethod
    def distance(p1, p2):
        """Calculate 2D distance between points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    @staticmethod
    def angle_wrap(angle):
        """Wrap angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def get_path_following_cmd(self):
        """Calculate control command for path following"""
        if not self.path_following or not self.current_path:
            return None
            
        if self.goal_idx >= len(self.current_path):
            rospy.loginfo("Path following completed!")
            self.path_following = False
            return None
        
        # Get current goal
        goal = self.current_path[self.goal_idx]
        current_pos = [self.robot_pose[0], self.robot_pose[1]]
        
        # Check if reached current waypoint
        dist_to_goal = self.distance(current_pos, goal[:2])
        if dist_to_goal < self.goal_tolerance:
            rospy.loginfo(f"Reached waypoint {self.goal_idx + 1}/{len(self.current_path)}")
            self.goal_idx += 1
            return self.get_path_following_cmd()  # Get command for next waypoint
        
        # Calculate control
        angle_to_goal = math.atan2(goal[1] - current_pos[1], goal[0] - current_pos[0])
        angle_error = self.angle_wrap(angle_to_goal - self.robot_pose[2])
        
        # Create high command
        cmd = HighCmd()
        cmd.head = [0xFE, 0xEF]
        cmd.levelFlag = 0xEE  # HIGHLEVEL
        cmd.mode = 2  # Walk mode
        cmd.gaitType = 1  # Trot
        cmd.reserve = 0
        
        # Control logic
        if abs(angle_error) > self.angle_tolerance:
            # Turn towards goal
            cmd.velocity = [0.1 * self.linear_speed, 0.0]
            cmd.yawSpeed = self.angular_speed * (1.0 if angle_error > 0 else -1.0)
        else:
            # Move forward
            cmd.velocity = [self.linear_speed, 0.0]
            cmd.yawSpeed = 0.5 * self.angular_speed * angle_error
        
        # Clamp values
        cmd.velocity[0] = max(-0.4, min(0.4, cmd.velocity[0]))
        cmd.yawSpeed = max(-0.6, min(0.6, cmd.yawSpeed))
        
        # Update estimated pose (simple integration)
        dt = 0.002  # 500Hz
        self.robot_pose[0] += cmd.velocity[0] * math.cos(self.robot_pose[2]) * dt
        self.robot_pose[1] += cmd.velocity[0] * math.sin(self.robot_pose[2]) * dt
        self.robot_pose[2] += cmd.yawSpeed * dt
        self.robot_pose[2] = self.angle_wrap(self.robot_pose[2])
        
        return cmd
    
    def get_default_cmd(self):
        """Get default standing command"""
        cmd = HighCmd()
        cmd.head = [0xFE, 0xEF]
        cmd.levelFlag = 0xEE  # HIGHLEVEL
        cmd.mode = 0  # Idle mode
        cmd.gaitType = 0
        cmd.speedLevel = 0
        cmd.footRaiseHeight = 0
        cmd.bodyHeight = 0
        cmd.euler = [0, 0, 0]
        cmd.velocity = [0.0, 0.0]
        cmd.yawSpeed = 0.0
        cmd.reserve = 0
        return cmd
    
    def run(self):
        """Main control loop"""
        rospy.loginfo("Starting pathfinding walker...")
        rospy.loginfo("Use RViz to set start (2D Pose Estimate) and goal (2D Nav Goal)")
        
        while not rospy.is_shutdown():
            self.motiontime += 2  # Increment
            
            # Get command based on current state
            if self.path_following:
                cmd = self.get_path_following_cmd()
                if cmd is None:
                    cmd = self.get_default_cmd()  # Stop when path complete
            else:
                cmd = self.get_default_cmd()  # Stand and wait for path
            
            # Publish command
            self.cmd_pub.publish(cmd)
            
            # Sleep at 500Hz 
            self.rate.sleep()

def main():
    try:
        walker = PathfindingWalker()
        walker.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Pathfinding walker shutting down...")
    except Exception as e:
        rospy.logerr(f"Error in pathfinding walker: {e}")

if __name__ == '__main__':
    main()