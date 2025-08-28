import numpy as np
import os
import csv
from datetime import datetime
from typing import List, Tuple, Optional
import json

class DynamicObstacle:
    
    def __init__(self, position: Tuple[float, float, float], name: str = "DynamicObstacle"):
        self.position = np.array(position)
        self.name = name
        self.active = True
    
    def update_position(self, new_position: Tuple[float, float, float]):
       
        self.position = np.array(new_position)

class DynamicRepulsionManager:

    
    def __init__(self, grid_generator, 
                 dynamic_influence_radius: float = 1.0,
                 dynamic_k: float = 1.0):
        
        self.grid_gen = grid_generator
        self.dynamic_influence_radius = dynamic_influence_radius
        self.dynamic_k = dynamic_k
        
        # Export settings
        self.export_settings = {
            'save_csv': True,
            'save_numpy': True,
            'texture_name': 'dynamicPotential',
            'output_folder': 'PotentialMaps',
            'csv_folder': 'potentials'
        }
        
        # Dynamic obstacles list
        self.dynamic_obstacles: List[DynamicObstacle] = []
        
        # Cache grid info
        if self.grid_gen is None:
            raise ValueError("[DynamicRepulsionManager] No GridGenerator found.")
        
        self.grid = None
        self.grid_size_x = 0
        self.grid_size_y = 0
        self._update_grid_cache()
    
    def _update_grid_cache(self):
        
        self.grid = self.grid_gen.get_grid()
        if self.grid is None:
            raise ValueError("[DynamicRepulsionManager] Grid is None")
        
        self.grid_size_x = len(self.grid)
        self.grid_size_y = len(self.grid[0]) if self.grid_size_x > 0 else 0
        
        print(f"[DynamicRepulsionManager] Grid size: {self.grid_size_x} x {self.grid_size_y}")
    
    def add_dynamic_obstacle(self, position: Tuple[float, float, float], name: str = "DynamicObstacle"):
      
        obstacle = DynamicObstacle(position, name)
        self.dynamic_obstacles.append(obstacle)
        print(f"[DynamicRepulsionManager] Added dynamic obstacle '{name}' at {position}")
        return obstacle
    
    def remove_dynamic_obstacle(self, obstacle: DynamicObstacle):
 
        if obstacle in self.dynamic_obstacles:
            self.dynamic_obstacles.remove(obstacle)
            print(f"[DynamicRepulsionManager] Removed dynamic obstacle '{obstacle.name}'")
    
    def clear_dynamic_obstacles(self):
      
        count = len(self.dynamic_obstacles)
        self.dynamic_obstacles.clear()
        print(f"[DynamicRepulsionManager] Cleared {count} dynamic obstacles")
    
    def update_obstacle_position(self, obstacle: DynamicObstacle, new_position: Tuple[float, float, float]):
       
        obstacle.update_position(new_position)
    
    def apply_dynamic_repulsion(self):

        if self.grid is None:
            print("[DynamicRepulsionManager] Grid is None, cannot apply repulsion")
            return
        
        # Zero out last frame's values
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                if self.grid[x][y] is not None:
                    self.grid[x][y].dynamic_repulsive_potential = 0.0
        
        # Collect active obstacle positions
        obstacle_positions = []
        for obs in self.dynamic_obstacles:
            if obs.active:
                obstacle_positions.append(obs.position)
        
        if not obstacle_positions:
            return  # No active obstacles
        
        # For each cell, sum contributions from all dynamic obstacles
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                node = self.grid[x][y]
                if node is None or not node.walkable:
                    continue
                
                world_pos = np.array(node.world_position)
                total_repulsion = 0.0
                
                for obs_pos in obstacle_positions:
                    # Calculate distance
                    distance = np.linalg.norm(world_pos - obs_pos)
                    
                    # Apply repulsion if within influence radius
                    if distance <= self.dynamic_influence_radius:
                        repulsion = self.dynamic_k * np.exp(-distance)
                        total_repulsion += repulsion
                
                node.dynamic_repulsive_potential = total_repulsion
    
    def update(self):
       
        self.apply_dynamic_repulsion()
    
    def get_dynamic_potential_field(self) -> np.ndarray:
       
        if self.grid is None:
            return np.zeros((0, 0))
        
        field = np.zeros((self.grid_size_x, self.grid_size_y), dtype=np.float32)
        
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                node = self.grid[x][y]
                if node is not None:
                    field[x, y] = node.dynamic_repulsive_potential
        
        return field
    
    def save_dynamic_potential(self, 
                             save_csv: bool = True, 
                             save_numpy: bool = True,
                             output_folder: str = None,
                             csv_folder: str = None):
       
        if self.grid is None:
            print("[DynamicRepulsionManager] No grid found - cannot export.")
            return
        
        # Get the potential field
        field = self.get_dynamic_potential_field()
        max_val = np.max(field) if field.size > 0 else 0.0
        
        # Use provided folders or defaults
        output_dir = output_folder or self.export_settings['output_folder']
        csv_dir = csv_folder or self.export_settings['csv_folder']
        texture_name = self.export_settings['texture_name']
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        
        # Save as NumPy array (.npy)
        if save_numpy:
            numpy_path = os.path.join(output_dir, f"{texture_name}.npy")
            np.save(numpy_path, field)
            print(f"[DynamicRepulsionManager] Saved NumPy array → {numpy_path}")
        
        # Save as CSV
        if save_csv:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            csv_filename = f"DynamicPotential_{self.grid_size_x}x{self.grid_size_y}_{timestamp}.csv"
            csv_path = os.path.join(csv_dir, csv_filename)
            
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header row
                header = ['Y\\X'] + [str(x) for x in range(self.grid_size_x)]
                writer.writerow(header)
                
                # Write data rows
                for y in range(self.grid_size_y):
                    row = [str(y)]
                    for x in range(self.grid_size_x):
                        row.append(f"{field[x, y]:.4f}")
                    writer.writerow(row)
            
            print(f"[DynamicRepulsionManager] Saved CSV: {csv_path}")
        
        # Save metadata as JSON
        metadata = {
            'grid_size': [self.grid_size_x, self.grid_size_y],
            'max_value': float(max_val),
            'dynamic_influence_radius': self.dynamic_influence_radius,
            'dynamic_k': self.dynamic_k,
            'num_obstacles': len([obs for obs in self.dynamic_obstacles if obs.active]),
            'timestamp': datetime.now().isoformat(),
            'obstacles': [
                {
                    'name': obs.name,
                    'position': obs.position.tolist(),
                    'active': obs.active
                }
                for obs in self.dynamic_obstacles
            ]
        }
        
        metadata_path = os.path.join(output_dir, f"{texture_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[DynamicRepulsionManager] Saved metadata → {metadata_path}")
    
    def get_obstacle_count(self) -> int:
        
        return len([obs for obs in self.dynamic_obstacles if obs.active])
    
    def set_influence_radius(self, radius: float):
      
        self.dynamic_influence_radius = max(0.0, radius)
        print(f"[DynamicRepulsionManager] Set influence radius to {self.dynamic_influence_radius}")
    
    def set_dynamic_k(self, k_value: float):
       
        self.dynamic_k = max(0.0, k_value)
        print(f"[DynamicRepulsionManager] Set dynamic K to {self.dynamic_k}")
    
    def print_status(self):
       
        active_count = self.get_obstacle_count()
        print(f"[DynamicRepulsionManager] Status:")
        print(f"  Grid size: {self.grid_size_x} x {self.grid_size_y}")
        print(f"  Active obstacles: {active_count}")
        print(f"  Influence radius: {self.dynamic_influence_radius}")
        print(f"  Dynamic K: {self.dynamic_k}")
        
        if active_count > 0:
            field = self.get_dynamic_potential_field()
            max_potential = np.max(field)
            min_potential = np.min(field)
            print(f"  Potential range: {min_potential:.4f} to {max_potential:.4f}")