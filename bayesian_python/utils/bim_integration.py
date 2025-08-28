import json
import numpy as np
from fields.repulsive_field import RepulsiveObstacle


class BIMObstacle:
    """
    Wrapper class for BIM objects that bridges:
    1. BIM data (from extracted_objects.json)
    2. Bayesian system (needs .name, .repulsive_gain, .prob_coef)
    3. Repulsive field (needs closest_point method)
    """
    
    def __init__(self, name, location, dimensions, family_name=None, danger_coefficient=0.5):
        # Basic BIM data
        self.name = name
        self.location = np.array(location)
        self.dimensions = np.array(dimensions)
        self.family_name = family_name or name.lower()
        
        # For Bayesian system - these will be read/updated by BayesianRepulsionUpdater
        self.prob_coef = danger_coefficient       # Prior probability (0.0 to 1.0)
        name_lower = name.lower()
        if 'wall' in name_lower:
            self.repulsive_gain = 0.1           
        elif 'workstation' in name_lower:
            self.repulsive_gain = 0.2            
        else:
            self.repulsive_gain = 0.1            
        
        # For repulsive field system
        self.max_influence_distance = 0.6     # How far this obstacle affects nodes
        
        # Compute bounding box for closest_point calculations
        self._compute_bounds()
    
    def _compute_bounds(self):
       
        half_dims = self.dimensions / 2.0
        self.min_bounds = self.location - half_dims
        self.max_bounds = self.location + half_dims
    
    def closest_point(self, position):

        pos = np.array(position)
        
        # Clamp position to bounding box (simple box collision)
        closest = np.clip(pos, self.min_bounds, self.max_bounds)
        
        return closest
    
    def get_danger_category(self):

        name_lower = self.name.lower()
        
        if 'wall' in name_lower:
            return 'structural'
        elif 'workstation' in name_lower:
            return 'work_area'
        elif any(keyword in name_lower for keyword in ['agent', 'capsule']):
            return 'general_obstacle''workstation'
        else:
            return 'structural'  # Most BIM elements are structural


class BIMIntegration:
    
    
    def __init__(self, bim_json_path):
        self.bim_json_path = bim_json_path
        self.bim_obstacles = []
        self.repulsive_obstacles = []
            
    def load_bim_obstacles(self):

        print(f"Loading BIM data from {self.bim_json_path}")
        
        with open(self.bim_json_path, 'r') as f:
            bim_data = json.load(f)
        
        self.bim_obstacles = []
        skipped_count = 0
        
        for obj_name, obj_data in bim_data.items():
            # Skip navigation waypoints and physical equipment
            name_lower = obj_name.lower()
            if any(skip_name in name_lower for skip_name in ['agent', 'capsule']):
                print(f"  Skipping navigation/equipment object: {obj_name}")
                skipped_count += 1
                continue
                
            # Only process MESH objects (skip EMPTY objects)
            if obj_data.get('type') != 'MESH':
                skipped_count += 1
                continue
                
            # Skip objects that are too small (might be details)
            dimensions = obj_data.get('dimensions', [0, 0, 0])
            if max(dimensions) < 0.1:  # Skip objects smaller than 10cm
                skipped_count += 1
                continue
            
            # Create BIMObstacle wrapper for objects suitable for Bayesian analysis
            obstacle = BIMObstacle(
                name=obj_name,
                location=obj_data['location'],
                dimensions=obj_data['dimensions'],
                family_name=obj_data.get('family_name'),
                danger_coefficient=self._get_default_danger(obj_name)
            )
            
            self.bim_obstacles.append(obstacle)
            
        print(f"Loaded {len(self.bim_obstacles)} BIM obstacles for Bayesian analysis")
        print(f"Skipped {skipped_count} objects (navigation waypoints, equipment, or invalid)")
        return self.bim_obstacles
        
    def _get_default_danger(self, obj_name):

        name_lower = obj_name.lower()
        if 'wall' in name_lower:
            return 0.5  
        elif 'workstation' in name_lower:
            return 0.5 
        elif any(keyword in name_lower for keyword in ['agent', 'capsule']):
            return 0.5 
        else:
            return 0.5  
    
    def create_repulsive_obstacles(self):

        self.repulsive_obstacles = []
        
        for bim_obs in self.bim_obstacles:
            # Create RepulsiveObstacle with a closure that captures the BIMObstacle
            repulsive_obs = RepulsiveObstacle(
                get_closest_point_fn=lambda pos, obs=bim_obs: obs.closest_point(pos),
                gain=bim_obs.repulsive_gain,
                max_influence=bim_obs.max_influence_distance
            )
            
            # Keep reference to BIM obstacle for updates
            repulsive_obs._bim_obstacle = bim_obs
            
            self.repulsive_obstacles.append(repulsive_obs)
        
        print(f"Created {len(self.repulsive_obstacles)} repulsive obstacles")
        return self.repulsive_obstacles
    
    def update_repulsive_gains(self):
       
        for rep_obs in self.repulsive_obstacles:
            if hasattr(rep_obs, '_bim_obstacle'):
                rep_obs.gain = rep_obs._bim_obstacle.repulsive_gain
        
        print("Updated repulsive obstacle gains from Bayesian results")
    
    def get_obstacles_for_bayesian(self):

        return self.bim_obstacles
    
    def get_obstacles_for_repulsive_field(self):

        return self.repulsive_obstacles
    
    def print_obstacle_summary(self):
 
        print("\nBIM OBSTACLE SUMMARY:")
        categories = {}
        
        for obs in self.bim_obstacles:
            category = obs.get_danger_category()
            if category not in categories:
                categories[category] = []
            categories[category].append(obs.name)
        
        for category, names in categories.items():
            print(f"{category}: {len(names)} objects")
            for name in names[:3]:  # Show first 3 examples
                print(f"  - {name}")
            if len(names) > 3:
                print(f"  ... and {len(names) - 3} more")
        
        print(f"\nTotal obstacles: {len(self.bim_obstacles)}")


def create_complete_integration(bim_json_path):
   
    integration = BIMIntegration(bim_json_path)
    integration.load_bim_obstacles()
    integration.create_repulsive_obstacles()
    integration.print_obstacle_summary()
    
    return integration