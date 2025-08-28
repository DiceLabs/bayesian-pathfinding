import numpy as np
import open3d as o3d
import math
import open3d.core as o3c
from collections import deque
from .node import Node
class GridGenerator:
    def __init__(self, mesh_file, cell_size=0.025, ray_height=0.7):
       
        self.mesh = o3d.io.read_triangle_mesh(mesh_file)
        self.mesh.compute_vertex_normals()
        
       
        verts = np.asarray(self.mesh.vertices)
        self.min_b, self.max_b = verts.min(axis=0), verts.max(axis=0)
        
       
        z_values = np.asarray(self.mesh.vertices)[:,2]
        sorted_z = np.sort(z_values)
        threshold_index = int(len(sorted_z) * 0.1)  #use the lowest 10%
        self.base_z = np.median(sorted_z[:threshold_index])  
        self.cell_size = cell_size
        self.node_diameter = cell_size * 2
        self.ray_height = ray_height

  
        self.grid_size_x = int(round((self.max_b[0] - self.min_b[0]) / self.node_diameter))
        self.grid_size_y = int(round((self.max_b[1] - self.min_b[1]) / self.node_diameter))
        print(f"Grid Size: {self.grid_size_x} x {self.grid_size_y} with Node Diameter: {self.node_diameter}")

     
        self.grid = [[None for _ in range(self.grid_size_y)] for _ in range(self.grid_size_x)]
        
        
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
    
    
    def generate_grid(self):


        print("Generating Grid...")
        walkable_points = []
        obstacle_points =[]

        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                world_x = self.min_b[0] + x * self.node_diameter
                world_y = self.min_b[1] + y * self.node_diameter
                world_point = [world_x, world_y, self.base_z]

                #check for Obstacles (Raycasting)
                is_obstacle = self.is_obstacle(world_point)
                walkable = not is_obstacle

                node = Node(x, y, walkable, world_position=np.array(world_point))
                self.grid[x][y] = node

                if walkable:
                    walkable_points.append(world_point)
                else:
                    obstacle_points.append(world_point)
            
        self.assign_neighbors()  #assign neighbors after grid generation

        return np.array(walkable_points), np.array(obstacle_points)

    
    def is_obstacle(self, point):
    
        #start the ray just above our floor‐grid
        ray_start = [point[0], point[1], self.base_z + self.ray_height]
        rays = o3c.Tensor([ray_start + [0, 0, -1]], dtype=o3c.Dtype.Float32)
        result = self.scene.cast_rays(rays)
        t_hit = result['t_hit'][0].item()

        # no hit so  walkable
        if t_hit == np.inf:
            return False

        hit_z = ray_start[2] - t_hit

 
        if hit_z <= self.base_z + 1e-4:
            return False

     
        return True


    def assign_neighbors(self):
  
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                node = self.grid[x][y]
                if node and node.walkable:
                    node.neighbors = self.get_immediate_neighbors(node)

    def get_immediate_neighbors(self,node):

        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        nbrs = []
        for dx, dy in dirs:
            nx, ny = node.x + dx, node.y + dy
            if not (0 <= nx < self.grid_size_x and 0 <= ny < self.grid_size_y):
                continue

            cand = self.grid[nx][ny]
            if not cand.walkable:
                continue

           
            if dx != 0 and dy != 0:
                straight1 = self.grid[node.x + dx][node.y]
                straight2 = self.grid[node.x][node.y + dy]
                if not (straight1.walkable and straight2.walkable):
                    continue

            nbrs.append(cand)


        return nbrs





    def node_from_world_point(self,world_position):
     


        percent_x = (world_position[0] - self.min_b[0]) /(self.max_b[0] - self.min_b[0])
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
            print(f"Closest node found at {closest_node.world_position} (walkable)")
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
                    
                    if neighbor not in visited:
                        if neighbor.walkable:
                            # Calculate distance from original position
                            distance = np.linalg.norm(np.array(neighbor.world_position) - np.array(world_position))
                            if distance < best_distance:
                                best_distance = distance
                                best_node = neighbor
                        
                        nodes_to_check.append(neighbor)
                        visited.add(neighbor)

        if best_node:
            print(f"Closest walkable neighbor found at {best_node.world_position} (distance: {best_distance:.3f}m)")
            return best_node
        
        print(f"No walkable node found in vicinity of {world_position}")
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
                if neighbor:
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
        print(f"Total walkable nodes: {walkables} / {self.grid_size_x * self.grid_size_y}")

    @property
    def nodes(self):
        return self.grid

    @property
    def height(self):
        return self.grid_size_y

    @property
    def width(self):
        return self.grid_size_x
