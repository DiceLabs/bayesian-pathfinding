import numpy as np
import open3d as o3d
import math
import open3d.core as o3c
from collections import deque
from .node import Node
import rospy

class GridGenerator:
    def __init__(self, mesh_file, cell_size=0.025, robot_radius=0.3, margin=0.1, max_detect_z=2.0):
        """
        Modified constructor to accept ROS parameters while keeping your implementation
        """
        self.mesh = o3d.io.read_triangle_mesh(mesh_file)
        self.mesh.compute_vertex_normals()
        
        verts = np.asarray(self.mesh.vertices)
        self.min_b, self.max_b = verts.min(axis=0), verts.max(axis=0)
        
        z_values = np.asarray(self.mesh.vertices)[:,2]
        sorted_z = np.sort(z_values)
        threshold_index = int(len(sorted_z) * 0.1)  # use the lowest 10%
        self.base_z = sorted_z[0]  # Use the absolute lowest point
        self.cell_size = cell_size
        self.node_diameter = cell_size * 2
        
        # Map ROS parameter to your ray_height (or use default)
        self.ray_height = max_detect_z if max_detect_z > 0 else 0.7
        
        # Store ROS parameters for compatibility
        self.robot_radius = robot_radius
        self.margin = margin
        self.max_detect_z = max_detect_z

        self.grid_size_x = int(round((self.max_b[0] - self.min_b[0]) / self.node_diameter))
        self.grid_size_y = int(round((self.max_b[1] - self.min_b[1]) / self.node_diameter))
        rospy.loginfo(f"Grid Size: {self.grid_size_x} x {self.grid_size_y} with Node Diameter: {self.node_diameter}")

        self.grid = [[None for _ in range(self.grid_size_y)] for _ in range(self.grid_size_x)]
        
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))

    def generate(self):
        """
        ROS compatibility method - calls your generate_grid method
        """
        return self.generate_grid()
    
    def generate_grid(self):
        """Your original implementation with SciPy configuration space"""
        rospy.loginfo("Generating Grid...")
        walkable_points = []
        obstacle_points = []

        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                world_x = self.min_b[0] + x * self.node_diameter
                world_y = self.min_b[1] + y * self.node_diameter
                world_point = [world_x, world_y, self.base_z]

                # check for Obstacles (Raycasting)
                is_obstacle = self.is_obstacle(world_point)
                walkable = not is_obstacle

                node = Node(x, y, walkable, world_position=np.array(world_point))
                self.grid[x][y] = node

                if walkable:
                    walkable_points.append(world_point)
                else:
                    obstacle_points.append(world_point)
        
        # Apply configuration space transformation using SciPy method
        if self.robot_radius > 0:
            total_radius = self.robot_radius + self.margin
            inflation_cells = int(math.ceil(total_radius / self.node_diameter))
            self.inflate_obstacles(inflation_cells)
            rospy.loginfo(f"Configuration space applied using SciPy inflation: {inflation_cells} cells")
        
        # Recalculate walkable/obstacle points after C-space
        walkable_points = []
        obstacle_points = []
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                node = self.grid[x][y]
                if node:
                    if node.walkable:
                        walkable_points.append(node.world_position)
                    else:
                        obstacle_points.append(node.world_position)
        
        self.assign_neighbors()  # assign neighbors after C-space transformation
        
        return np.array(walkable_points), np.array(obstacle_points)
    
    def is_obstacle(self, point):
        # start the ray just above our floor‐grid
        ray_start = [point[0], point[1], self.base_z + self.ray_height]
        rays = o3c.Tensor([ray_start + [0, 0, -1]], dtype=o3c.Dtype.Float32)
        result = self.scene.cast_rays(rays)
        t_hit = result['t_hit'][0].item()

        # no hit so walkable
        if t_hit == np.inf:
            return False

        hit_z = ray_start[2] - t_hit

        if hit_z <= self.base_z + 1e-4:
            return False

        return True

    def apply_configuration_space(self, robot_radius=None, safety_margin=None):
        """
        Apply configuration space transformation by growing obstacles
        """
        if robot_radius is None:
            robot_radius = self.robot_radius
        if safety_margin is None:
            safety_margin = self.margin
        
        total_radius = robot_radius + safety_margin

        grid_radius = int(np.ceil(total_radius / self.node_diameter))
        
        rospy.loginfo(f"Applying C-space: robot_radius={robot_radius:.3f}m, safety_margin={safety_margin:.3f}m")
        rospy.loginfo(f"Total radius={total_radius:.3f}m, grid_radius={grid_radius} cells")
        
        # Create copy of original walkability
        original_grid = [[self.grid[x][y].walkable if self.grid[x][y] else False 
                        for y in range(self.grid_size_y)] 
                        for x in range(self.grid_size_x)]
        
        # Find all obstacle cells
        obstacle_cells = []
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                if self.grid[x][y] and not self.grid[x][y].walkable:
                    obstacle_cells.append((x, y))
        
        # Grow obstacles by robot radius + margin
        newly_blocked = 0
        for obs_x, obs_y in obstacle_cells:
            for dx in range(-grid_radius, grid_radius + 1):
                for dy in range(-grid_radius, grid_radius + 1):
                    # Check if within circular radius
                    distance = np.sqrt(dx*dx + dy*dy) * self.node_diameter
                    if distance <= total_radius:
                        new_x, new_y = obs_x + dx, obs_y + dy
                        
                        # Check bounds
                        if (0 <= new_x < self.grid_size_x and 
                            0 <= new_y < self.grid_size_y and
                            self.grid[new_x][new_y]):
                            
                            # Mark as obstacle in C-space
                            if original_grid[new_x][new_y]:  # Was originally walkable
                                self.grid[new_x][new_y].walkable = False
                                newly_blocked += 1
        
        # Reassign neighbors after C-space transformation
        self.assign_neighbors()
        
        rospy.loginfo(f"C-space applied: {newly_blocked} cells newly blocked")
        return newly_blocked

    def inflate_obstacles(self, inflation_radius_cells):
        """
        Apply configuration space transformation using SciPy morphological dilation
        """
        import numpy as np
        from scipy.ndimage import grey_dilation
        
        rospy.loginfo(f"Inflating obstacles by {inflation_radius_cells} cells using SciPy...")
        
        # Create binary grid: 1 = obstacle, 0 = free
        binary_grid = np.zeros((self.grid_size_x, self.grid_size_y), dtype=np.uint8)
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                if not self.grid[x][y].walkable:
                    binary_grid[x][y] = 1
        
        # Create square structuring element
        r = inflation_radius_cells
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        struct = ((xx*xx + yy*yy) <= r*r).astype(np.uint8)

        # Apply morphological dilation
        inflated = grey_dilation(binary_grid, footprint=struct)
        
        # Update grid with inflated obstacles
        newly_blocked = 0
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                if inflated[x][y] == 1 and self.grid[x][y].walkable:
                    self.grid[x][y].walkable = False
                    newly_blocked += 1
        
        rospy.loginfo(f"SciPy C-space applied: {newly_blocked} additional cells blocked")




    def assign_neighbors(self):
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                node = self.grid[x][y]
                if node and node.walkable:
                    node.neighbors = self.get_immediate_neighbors(node)

    def get_immediate_neighbors(self, node):
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        nbrs = []
        for dx, dy in dirs:
            nx, ny = node.x + dx, node.y + dy
            if not (0 <= nx < self.grid_size_x and 0 <= ny < self.grid_size_y):
                continue

            cand = self.grid[nx][ny]
            if not cand or not cand.walkable:  # Added null check
                continue

            if dx != 0 and dy != 0:
                straight1 = self.grid[node.x + dx][node.y]
                straight2 = self.grid[node.x][node.y + dy]
                if not (straight1 and straight1.walkable and straight2 and straight2.walkable):  # Added null checks
                    continue

            nbrs.append(cand)

        return nbrs

    def node_from_world_point(self, world_position):
        percent_x = (world_position[0] - self.min_b[0]) / (self.max_b[0] - self.min_b[0])
        percent_y = (world_position[1] - self.min_b[1]) / (self.max_b[1] - self.min_b[1])
        percent_x = max(0.0, min(1.0, percent_x))
        percent_y = max(0.0, min(1.0, percent_y))

        x = int(round((self.grid_size_x - 1) * percent_x))
        y = int(round((self.grid_size_y - 1) * percent_y))

        return self.grid[x][y] 

    def closest_walkable_node(self, world_position, allow_unwalkable=False):
        closest_node = self.node_from_world_point(world_position)
        
        if not closest_node:
            return None
            
        if closest_node.walkable:
            rospy.loginfo(f"Closest node found at {closest_node.world_position} (walkable)")
            return closest_node

        # Find the CLOSEST walkable node by distance, not just the first one
        nodes_to_check = deque([closest_node])
        visited = set([closest_node])
        best_node = None
        best_distance = float('inf')

        while nodes_to_check:
            current_node = nodes_to_check.popleft()
            
            # Check all 8 immediate grid neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    check_x = current_node.x + dx
                    check_y = current_node.y + dy
                    
                    if not (0 <= check_x < self.grid_size_x and 0 <= check_y < self.grid_size_y):
                        continue
                    
                    neighbor = self.grid[check_x][check_y]
                    
                    if neighbor and neighbor not in visited:  # Added null check
                        if neighbor.walkable:
                            # Calculate distance from original position
                            distance = np.linalg.norm(np.array(neighbor.world_position) - np.array(world_position))
                            if distance < best_distance:
                                best_distance = distance
                                best_node = neighbor
                        
                        nodes_to_check.append(neighbor)
                        visited.add(neighbor)

        if best_node:
            rospy.loginfo(f"Closest walkable neighbor found at {best_node.world_position} (distance: {best_distance:.3f}m)")
            return best_node
        
        rospy.logwarn(f"No walkable node found in vicinity of {world_position}")
        return None
    
    def mark_path_nodes(self, path_nodes):
        for node in path_nodes:
            node.is_path = True
   
    def update_node_position(self, node, new_position):
        node.world_position = np.array(new_position)
        node.g = float('inf')
        node.h = float('inf')
        node.parent = None

    def get_neighbors(self, node, radius=1):
        neighbors = []
        start_x = max(0, node.x - radius)
        end_x = min(self.grid_size_x - 1, node.x + radius)
        start_y = max(0, node.y - radius)
        end_y = min(self.grid_size_y - 1, node.y + radius)

        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                if x == node.x and y == node.y:
                    continue  # skip the center node itself
                neighbor = self.grid[x][y]
                if neighbor:  # Added null check
                    neighbors.append(neighbor)

        return neighbors

    def get_all_nodes(self):
        return [node for row in self.grid for node in row if node]

    def set_expanded_nodes(self, nodes):
        self.expanded_nodes = set(nodes)

    def set_goal_node(self, node):
        self.goal_node = node

    def get_grid(self):
        return self.grid

    def print_grid_summary(self):
        walkables = sum(1 for row in self.grid for n in row if n and n.walkable)
        rospy.loginfo(f"Total walkable nodes: {walkables} / {self.grid_size_x * self.grid_size_y}")

    @property
    def nodes(self):
        return self.grid

    @property
    def height(self):
        return self.grid_size_y

    @property
    def width(self):
        return self.grid_size_x