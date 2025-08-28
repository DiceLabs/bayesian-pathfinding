#!/usr/bin/env python3
import os
import json
import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray 
from std_msgs.msg import Header, ColorRGBA

# Import your new classes
from pathfinding_simulation.grid.grid_generator_open3d import GridGenerator
from pathfinding_simulation.grid.node import Node
from pathfinding_simulation.pathfinding.mhastar import MultiHeuristicAStar
from pathfinding_simulation.fields.repulsive_field import compute_repulsive_field
from pathfinding_simulation.utils.bim_integration import create_complete_integration
from pathfinding_simulation.utils.bayesian_repulsion_updater import BayesianRepulsionUpdater

import rospkg

class PathfindingNode:
    def __init__(self):
        rospy.init_node("pathfinding_node")

        api_key = rospy.get_param("~api_key", "")
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("pathfinding_simulation")
        
        # Mesh and parameters
                
        mesh_path = rospy.get_param("~mesh_file")
        cell_size = rospy.get_param("~cell_size", 0.025)
        ray_height = rospy.get_param("~ray_height", 0.7)

        # Initialize grid generator with new class
        # Go1-specific robot parameters

        self.grid_gen = GridGenerator(
            mesh_file=mesh_path,
            cell_size=cell_size,
            robot_radius=rospy.get_param("~robot_radius", 0.14),  # Use launch file value
            margin=rospy.get_param("~safety_margin", 0.10),      
            max_detect_z=rospy.get_param("~max_detect_z", 2.0)
        )

        # Generate the grid
        walkable_points, obstacle_points = self.grid_gen.generate_grid()
        rospy.loginfo(f"Grid generated: {len(walkable_points)} walkable, {len(obstacle_points)} obstacles")
        rospy.loginfo(f"Grid size: {self.grid_gen.grid_size_x} × {self.grid_gen.grid_size_y}")

        # BIM Integration for obstacles and repulsive fields
        extracted_json = os.path.join(pkg_path, "models/dicelab_bim/meshes/extracted_objects.json")
        if os.path.exists(extracted_json):
            self.bim_integration = create_complete_integration(extracted_json)
            
            # Setup Bayesian updater if API key provided
            if api_key.strip():
                self.bayesian_updater = BayesianRepulsionUpdater(
                    api_key=api_key,
                    obstacles=self.bim_integration.get_obstacles_for_bayesian()
                )
            else:
                self.bayesian_updater = None
                rospy.logwarn("No API key provided - skipping Bayesian updates")
            
            # Compute initial repulsive field
            repulsive_obstacles = self.bim_integration.get_obstacles_for_repulsive_field()
            compute_repulsive_field(
                grid=self.grid_gen.get_grid(),
                obstacles=repulsive_obstacles
            )
            rospy.loginfo("Repulsive field computed")
        else:
            rospy.logwarn(f"BIM data not found at {extracted_json}")
            self.bim_integration = None
            self.bayesian_updater = None

        # Publishers
        self.path_pub = rospy.Publisher("mha_path", Path, queue_size=1, latch=True)
        self.vis_pub = rospy.Publisher("mha_path_vis", Marker, queue_size=1, latch=True)
        self.grid_marker_pub = rospy.Publisher("/grid_markers", MarkerArray, queue_size=1, latch=True)

        # Subscribe to RViz tools
        self.start = None
        self.goal = None
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.cb_start)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.cb_goal)

        rospy.loginfo("Pathfinding node ready. Use RViz to set start (2D Pose Estimate) and goal (2D Nav Goal).")
        self.publish_grid()

    def publish_grid(self):
        """Publish grid visualization markers"""
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
                    # Color intensity based on repulsive potential
                    intensity = min(1.0, node.repulsive_potential * 0.5)
                    marker.color = ColorRGBA(r=intensity, g=1.0-intensity, b=0.0, a=0.3)
                else:
                    marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)  # red for obstacles

                marker.lifetime = rospy.Duration(0)
                marker_array.markers.append(marker)
                marker_id += 1

        rospy.loginfo(f"Total markers published: {marker_id}")
        self.grid_marker_pub.publish(marker_array)

    def cb_start(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        self.start = np.array([p.x, p.y, p.z])
        rospy.loginfo(f"Start set to {self.start}")
        self.try_plan()

    def cb_goal(self, msg: PoseStamped):
        p = msg.pose.position
        self.goal = np.array([p.x, p.y, p.z])
        rospy.loginfo(f"Goal set to {self.goal}")
        self.try_plan()

    def try_plan(self):
        if self.start is None or self.goal is None:
            return

        # Debug start and goal mapping
        rospy.loginfo(f"[DEBUG] Planning from {self.start} to {self.goal}")
        
        start_node = self.grid_gen.closest_walkable_node(self.start)
        goal_node = self.grid_gen.closest_walkable_node(self.goal)

        if not start_node or not goal_node:
            rospy.logwarn("[DEBUG] Could not map start/goal to walkable nodes")
            return

        rospy.loginfo(f"[DEBUG] Start node: ({start_node.x}, {start_node.y}) walkable: {start_node.walkable}")
        rospy.loginfo(f"[DEBUG] Goal node: ({goal_node.x}, {goal_node.y}) walkable: {goal_node.walkable}")

        # Perform Bayesian update if available
        if self.bayesian_updater:
            rospy.loginfo("Performing Bayesian update...")
            self.bayesian_updater.perform_update()
            
            # Update repulsive field with new gains
            if self.bim_integration:
                self.bim_integration.update_repulsive_gains()
                repulsive_obstacles = self.bim_integration.get_obstacles_for_repulsive_field()
                compute_repulsive_field(
                    grid=self.grid_gen.get_grid(),
                    obstacles=repulsive_obstacles
                )

        # Initialize Multi-Heuristic A* pathfinder
        mha_star = MultiHeuristicAStar(
            grid_generator=self.grid_gen,
            start_pos=self.start,
            goal_pos=self.goal,
            w1=rospy.get_param("~w1", 1.0),
            w2=rospy.get_param("~w2", 2.0),
            gamma=rospy.get_param("~gamma", 5.0)
        )

        # Plan path using your new algorithm
        try:
            # Get BIM obstacles for distance metrics
            obstacles = None
            if self.bim_integration:
                obstacles = self.bim_integration.get_obstacles_for_repulsive_field()
            
            # Plan the path
            mha_star.plan_path(
                start_position=self.start,
                goal_position=self.goal,
                bayes_manager=self.bayesian_updater,
                obstacles=obstacles,
                visualize_fn=self.visualize_path_callback
            )
            
        except Exception as e:
            rospy.logerr(f"[DEBUG] Path planning failed: {str(e)}")
            return

    def visualize_path_callback(self, path_nodes):
        """Callback for path visualization from MHA* planner"""
        if path_nodes:
            self.publish_path(path_nodes)
        else:
            rospy.logwarn("No path found")

    def publish_path(self, path_nodes):
        """Publish the path for RViz visualization and path following"""
        if not path_nodes:
            return
        
        hdr = Header(stamp=rospy.Time.now(), frame_id="map")
        path_msg = Path(header=hdr)
        
        for node in path_nodes:
            ps = PoseStamped(header=hdr)
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = node.world_position
            path_msg.poses.append(ps)
        
        self.path_pub.publish(path_msg)

        # Create visualization marker
        mk = Marker(header=hdr, ns="path", id=0, type=Marker.LINE_STRIP, action=Marker.ADD)
        mk.scale.x = 0.05
        mk.color = ColorRGBA(r=0.2, g=0.2, b=1.0, a=1.0)
        
        for node in path_nodes:
            x, y, z = node.world_position
            mk.points.append(Point(x, y, z + 0.05))
        
        self.vis_pub.publish(mk)

        rospy.loginfo(f"Published path with {len(path_nodes)} nodes.")

    def spin(self):
        rospy.spin()

def main():
    """Entry point for rosrun/roslaunch wrapper."""
    try:
        node = PathfindingNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Failed to initialize pathfinding node: {str(e)}")