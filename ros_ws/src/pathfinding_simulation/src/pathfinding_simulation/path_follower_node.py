#!/usr/bin/env python3
import rospy
import math
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped, Point, TransformStamped
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker, MarkerArray 
from std_msgs.msg import Header, ColorRGBA
import tf2_ros
import tf2_geometry_msgs


class PathFollower:
    def __init__(self):
        rospy.init_node("path_follower_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribe to the new Multi-Heuristic A* path
        self.path_sub = rospy.Subscriber("/mha_path", Path, self.path_cb)
        
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        
        # Add grid visualization publisher
        self.grid_vis_pub = rospy.Publisher("/path_follower_grid", MarkerArray, queue_size=1, latch=True)

        # TF frame parameters - make configurable to match robot
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.odom_frame = rospy.get_param("~odom_frame", "odom") 
        self.map_frame = rospy.get_param("~map_frame", "map")
        
        # Path following parameters
        self.linear_speed = rospy.get_param("~linear_speed", 0.3)
        self.angular_speed = rospy.get_param("~angular_speed", 0.5)
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.15)
        self.angle_tolerance = rospy.get_param("~angle_tolerance", 0.1)
        
        # Grid visualization parameters
        self.show_grid = rospy.get_param("~show_grid", True)
        self.grid_alpha = rospy.get_param("~grid_alpha", 0.3)
        
        self.path = []
        self.curr_pose = None
        self.goal_idx = 0
        self.path_ready = False
        self.reached_goal = False
        self.path_start_time = None
        self.grid_data = None
        self.current_path_positions = []

        self.rate = rospy.Rate(10)
        
        # TF debugging
        self.last_tf_error_time = rospy.Time(0)
        self.tf_error_count = 0
        
        # Check TF tree health at startup
        self.check_tf_tree_health()
        
        # Try to get grid data from pathfinding node (if available)
        self.grid_subscriber = None
        self.request_grid_data()

    def check_tf_tree_health(self):
        """Check TF tree for common issues"""
        rospy.loginfo(f"[PathFollower] Checking TF tree health...")
        rospy.loginfo(f"[PathFollower] Using frames: {self.map_frame} -> {self.odom_frame} -> {self.base_frame}")
        
        # Wait a bit for TF to populate
        rospy.sleep(2.0)
        
        try:
            # Check if basic transforms exist
            rospy.loginfo("[PathFollower] Checking available transforms...")
            
            # Test map->odom
            try:
                map_to_odom = self.tf_buffer.lookup_transform(
                    self.map_frame, self.odom_frame, rospy.Time(0), rospy.Duration(1.0)
                )
                rospy.loginfo(f"[PathFollower] {self.map_frame}->{self.odom_frame} transform OK")
            except Exception as e:
                rospy.logwarn(f"[PathFollower] {self.map_frame}->{self.odom_frame} transform failed: {e}")
            
            # Test odom->base
            try:
                odom_to_base = self.tf_buffer.lookup_transform(
                    self.odom_frame, self.base_frame, rospy.Time(0), rospy.Duration(1.0)
                )
                rospy.loginfo(f"[PathFollower] {self.odom_frame}->{self.base_frame} transform OK")
            except Exception as e:
                rospy.logwarn(f"[PathFollower] {self.odom_frame}->{self.base_frame} transform failed: {e}")
            
            # Test full chain map->base
            try:
                map_to_base = self.tf_buffer.lookup_transform(
                    self.map_frame, self.base_frame, rospy.Time(0), rospy.Duration(1.0)
                )
                rospy.loginfo(f"[PathFollower] {self.map_frame}->{self.base_frame} full chain OK")
            except Exception as e:
                rospy.logwarn(f"[PathFollower] {self.map_frame}->{self.base_frame} full chain failed: {e}")
                
        except Exception as e:
            rospy.logwarn(f"[PathFollower] TF health check failed: {e}")

    def request_grid_data(self):
        """Request grid data from pathfinding node for visualization"""
        try:
            rospy.loginfo("[PathFollower] Waiting for grid data (this may take up to 30 seconds)...")
            grid_msg = rospy.wait_for_message("/grid_markers", MarkerArray, timeout=30.0)
            self.process_grid_data(grid_msg)
            rospy.loginfo("[PathFollower] Grid data received and processed")
        except rospy.ROSException:
            rospy.logwarn("[PathFollower] Could not get grid data from pathfinding node - will try to subscribe instead")
            self.setup_grid_subscriber()

    def setup_grid_subscriber(self):
        """Set up subscriber to catch grid data when it becomes available"""
        if self.grid_subscriber is None:
            rospy.loginfo("[PathFollower] Setting up grid data subscriber...")
            self.grid_subscriber = rospy.Subscriber("/grid_markers", MarkerArray, self.grid_data_callback, queue_size=1)
    
    def grid_data_callback(self, msg):
        """Callback for grid data subscriber"""
        rospy.loginfo("[PathFollower] Received grid data via subscriber")
        self.process_grid_data(msg)
        
        if self.grid_subscriber:
            self.grid_subscriber.unregister()
            self.grid_subscriber = None
            rospy.loginfo("[PathFollower] Grid subscriber unregistered")
        
        if self.show_grid:
            self.publish_grid_visualization()

    def process_grid_data(self, grid_msg):
        """Process grid data from pathfinding node and store for our visualization"""
        self.grid_data = []
        
        for marker in grid_msg.markers:
            grid_cell = {
                'position': [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],
                'scale': [marker.scale.x, marker.scale.y, marker.scale.z],
                'walkable': marker.color.r < 0.5
            }
            self.grid_data.append(grid_cell)
        
        rospy.loginfo(f"[PathFollower] Processed {len(self.grid_data)} grid cells")

    def try_get_grid_data_again(self):
        """Try to get grid data again"""
        try:
            rospy.loginfo("[PathFollower] Trying to get grid data again...")
            grid_msg = rospy.wait_for_message("/grid_markers", MarkerArray, timeout=2.0)
            self.process_grid_data(grid_msg)
            if self.show_grid:
                self.publish_grid_visualization()
            return True
        except rospy.ROSException:
            return False

    def publish_grid_visualization(self):
        """Publish grid visualization"""
        if not self.grid_data or not self.show_grid:
            return
            
        marker_array = MarkerArray()
        marker_id = 0
        
        path_positions = set()
        for pos in self.current_path_positions:
            rounded_pos = (round(pos[0], 3), round(pos[1], 3))
            path_positions.add(rounded_pos)
        
        for i, cell in enumerate(self.grid_data):
            marker = Marker()
            marker.header.frame_id = self.map_frame  # Use configurable frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "path_follower_grid"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = cell['position'][0]
            marker.pose.position.y = cell['position'][1]
            marker.pose.position.z = 0.01

            marker.scale.x = cell['scale'][0]
            marker.scale.y = cell['scale'][1]
            marker.scale.z = 0.02

            cell_pos = (round(cell['position'][0], 3), round(cell['position'][1], 3))
            is_on_path = cell_pos in path_positions

            if is_on_path:
                marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
                marker.scale.z = 0.05
            elif cell['walkable']:
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=self.grid_alpha)
            else:
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=self.grid_alpha)

            marker.lifetime = rospy.Duration(0)
            marker_array.markers.append(marker)
            marker_id += 1

        for i, pos in enumerate(self.current_path_positions):
            marker = Marker()
            marker.header.frame_id = self.map_frame  # Use configurable frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "path_waypoints"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = 0.1

            marker.scale.x = 0.08
            marker.scale.y = 0.08
            marker.scale.z = 0.08

            marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.9)
            marker.lifetime = rospy.Duration(0)
            marker_array.markers.append(marker)
            marker_id += 1

        self.grid_vis_pub.publish(marker_array)

    def path_cb(self, msg):
        """Callback for receiving new path from MHA* planner"""
        self.path = msg.poses
        self.goal_idx = 0
        self.path_ready = bool(self.path)
        self.reached_goal = False
        self.path_start_time = rospy.Time.now()
        
        self.current_path_positions = []
        for pose in self.path:
            pos = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            self.current_path_positions.append(pos)
        
        if not self.grid_data:
            if not self.try_get_grid_data_again():
                rospy.logwarn("[PathFollower] Still no grid data available")
        
        if self.show_grid and self.grid_data:
            self.publish_grid_visualization()
        
        if self.path_ready:
            rospy.loginfo(f"[PathFollower] Received new path with {len(self.path)} points.")
            if len(self.path) > 0:
                start = self.path[0].pose.position
                end = self.path[-1].pose.position
                rospy.loginfo(f"[PathFollower] Path: ({start.x:.2f}, {start.y:.2f}) -> ({end.x:.2f}, {end.y:.2f})")
        else:
            rospy.logwarn("[PathFollower] Received empty path.")
            self.cmd_pub.publish(Twist())

    def distance(self, p1, p2):
        """Calculate 2D distance between two points"""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def angle_difference(self, angle1, angle2):
        """Calculate the smallest angle difference between two angles"""
        diff = angle1 - angle2
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def get_robot_pose(self):
        """Get current robot pose from TF with better error handling"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rospy.Time(0), rospy.Duration(0.1)
            )
            
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # Convert quaternion to euler
            import tf.transformations
            quat = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            _, _, yaw = tf.transformations.euler_from_quaternion(quat)
            
            # Reset error count on success
            self.tf_error_count = 0
            
            return x, y, yaw
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            # Throttled logging to avoid spam
            current_time = rospy.Time.now()
            if (current_time - self.last_tf_error_time).to_sec() > 2.0:  # Log every 2 seconds max
                self.tf_error_count += 1
                rospy.logwarn(f"[PathFollower] TF lookup failed (#{self.tf_error_count}): {type(e).__name__}: {e}")
                rospy.logwarn(f"[PathFollower] Looking for transform: {self.map_frame} -> {self.base_frame}")
                self.last_tf_error_time = current_time
                
                # Suggest debugging steps
                if self.tf_error_count == 1:
                    rospy.logwarn("[PathFollower] TF Debug suggestions:")
                    rospy.logwarn("  1. Check: rosrun tf tf_echo map base")
                    rospy.logwarn("  2. Check: rostopic echo /tf")
                    rospy.logwarn("  3. Check: rosrun tf view_frames")
            
            return None, None, None
        except Exception as e:
            rospy.logwarn_throttle(5, f"[PathFollower] Unexpected TF error: {e}")
            return None, None, None

    def run(self):
        """Main path following loop"""
        rospy.loginfo("[PathFollower] Starting path following loop...")
        
        if self.show_grid and self.grid_data:
            self.publish_grid_visualization()
        
        while not rospy.is_shutdown():
            if not self.path_ready:
                self.cmd_pub.publish(Twist())
                self.rate.sleep()
                continue

            if self.goal_idx >= len(self.path):
                if not self.reached_goal:
                    duration = rospy.Time.now() - self.path_start_time
                    rospy.loginfo(f"[PathFollower] Reached final goal! Path completed in {duration.to_sec():.1f} seconds.")
                    self.reached_goal = True
                self.cmd_pub.publish(Twist())
                self.rate.sleep()
                continue

            curr_x, curr_y, curr_yaw = self.get_robot_pose()
            if curr_x is None:
                # Stop robot if we can't get pose
                self.cmd_pub.publish(Twist())
                self.rate.sleep()
                continue

            goal = self.path[self.goal_idx].pose.position
            dist_to_goal = self.distance(
                type('obj', (object,), {'x': curr_x, 'y': curr_y})(), 
                goal
            )

            if dist_to_goal < self.goal_tolerance:
                rospy.loginfo(f"[PathFollower] Reached waypoint {self.goal_idx + 1}/{len(self.path)}")
                self.goal_idx += 1
                continue

            angle_to_goal = math.atan2(goal.y - curr_y, goal.x - curr_x)
            angle_diff = self.angle_difference(angle_to_goal, curr_yaw)

            cmd = Twist()
            
            if abs(angle_diff) > self.angle_tolerance:
                cmd.angular.z = self.angular_speed * np.sign(angle_diff) * min(1.0, abs(angle_diff) / 0.5)
                cmd.linear.x = 0.1 * self.linear_speed
            else:
                cmd.linear.x = self.linear_speed
                cmd.angular.z = self.angular_speed * angle_diff

            rospy.loginfo_throttle(2, 
                f"[PathFollower] Pose: ({curr_x:.2f}, {curr_y:.2f}, {math.degrees(curr_yaw):.1f}°) "
                f"-> Goal {self.goal_idx + 1}: ({goal.x:.2f}, {goal.y:.2f}) "
                f"| Dist: {dist_to_goal:.2f}m, AngleDiff: {math.degrees(angle_diff):.1f}°"
            )

            self.cmd_pub.publish(cmd)
            self.rate.sleep()

def main():
    try:
        follower = PathFollower()
        follower.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("[PathFollower] Shutting down...")
    except Exception as e:
        rospy.logerr(f"[PathFollower] Error: {e}")