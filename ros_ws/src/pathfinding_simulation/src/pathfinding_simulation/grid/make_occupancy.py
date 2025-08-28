#!/usr/bin/env python3
import argparse
import os
import numpy as np
import imageio
import yaml
import builtins
import sys

# Mock rospy for standalone usage
class MockRospy:
    def loginfo(self, msg):
        print(f"[INFO] {msg}")
    
    def logwarn(self, msg):
        print(f"[WARN] {msg}")

# Replace rospy in grid_generator if running standalone
if 'rospy' not in sys.modules:
    sys.modules['rospy'] = MockRospy()

# Import your grid generator
from pathfinding_simulation.grid.grid_generator_open3d import GridGenerator

# Fix for numpy float serialization in YAML
yaml.SafeDumper.add_representer(np.float64, lambda dumper, value: dumper.represent_float(builtins.float(value)))
yaml.SafeDumper.add_representer(np.float32, lambda dumper, value: dumper.represent_float(builtins.float(value)))

def main():
    parser = argparse.ArgumentParser(description="Generate occupancy grid map from STL mesh")
    parser.add_argument("--input_mesh", required=True, help="Path to input STL mesh file")
    parser.add_argument("--output_pgm", required=True, help="Path to output PGM file")
    parser.add_argument("--output_yaml", required=True, help="Path to output YAML file")
    parser.add_argument("--cell_size", type=float, default=0.025, help="Grid cell size in meters")
    parser.add_argument("--robot_radius", type=float, default=0.14, help="Robot radius for configuration space")
    parser.add_argument("--safety_margin", type=float, default=0.06, help="Additional safety margin")
    parser.add_argument("--max_detect_z", type=float, default=1.5, help="Maximum obstacle detection height")
    
    args = parser.parse_args()
    
    print(f"Generating occupancy grid from: {args.input_mesh}")
    print(f"Output PGM: {args.output_pgm}")
    print(f"Output YAML: {args.output_yaml}")
    print(f"Cell size: {args.cell_size}m")
    print(f"Robot radius: {args.robot_radius}m")
    print(f"Safety margin: {args.safety_margin}m")
    
    # Initialize grid generator
    grid_gen = GridGenerator(
        mesh_file=args.input_mesh,
        cell_size=args.cell_size,
        robot_radius=args.robot_radius,
        margin=args.safety_margin,
        max_detect_z=args.max_detect_z
    )
    
    # Generate the grid (this applies configuration space automatically)
    walkable_points, obstacle_points = grid_gen.generate_grid()
    
    print(f"Grid generated: {len(walkable_points)} walkable, {len(obstacle_points)} obstacles")
    print(f"Grid size: {grid_gen.grid_size_x} x {grid_gen.grid_size_y}")
    
    # Get map dimensions
    map_width_m = grid_gen.max_b[0] - grid_gen.min_b[0]
    map_height_m = grid_gen.max_b[1] - grid_gen.min_b[1]
    
    print(f"Map Physical Dimensions:")
    print(f"Width: {map_width_m:.3f} meters")
    print(f"Height: {map_height_m:.3f} meters")
    print(f"Resolution: {args.cell_size} m/cell")
    
    # Create occupancy grid array (ROS format: 0=free, 100=occupied, -1=unknown)
    # Note: PGM format uses 0=black(occupied), 255=white(free)
    W, H = grid_gen.grid_size_x, grid_gen.grid_size_y
    occupancy_grid = np.zeros((H, W), dtype=np.uint8)
    
    # Fill the occupancy grid
    for x in range(W):
        for y in range(H):
            node = grid_gen.grid[x][y]
            if node is not None:
                if node.walkable:
                    occupancy_grid[y, x] = 254  # Free space (white in PGM)
                else:
                    occupancy_grid[y, x] = 0    # Occupied (black in PGM)
            else:
                occupancy_grid[y, x] = 205      # Unknown (gray in PGM)
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.output_pgm), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_yaml), exist_ok=True)
    occupancy_grid = np.flipud(occupancy_grid)
    # Save PGM file
    imageio.imwrite(args.output_pgm, occupancy_grid)
    print(f"Saved PGM file: {args.output_pgm}")
    
    # Create YAML metadata
    map_yaml = {
        'image': os.path.basename(args.output_pgm),
        'resolution': float(args.cell_size * 2),  # meters per pixel
        'origin': [float(grid_gen.min_b[0]), float(grid_gen.min_b[1]), 0.0],  # x, y, yaw
        'negate': 0,  # 0 = white=free, black=occupied
        'occupied_thresh': 0.65,  # Pixels with occupancy probability > this are occupied
        'free_thresh': 0.196,     # Pixels with occupancy probability < this are free
    }
    
    # Save YAML file
    with open(args.output_yaml, 'w') as f:
        yaml.dump(map_yaml, f, Dumper=yaml.SafeDumper, default_flow_style=False)
    
    print(f"Saved YAML file: {args.output_yaml}")
    
    # Print summary
    total_cells = W * H
    free_cells = np.sum(occupancy_grid == 254)
    occupied_cells = np.sum(occupancy_grid == 0)
    unknown_cells = np.sum(occupancy_grid == 205)
    
    print(f"\nMap Statistics:")
    print(f"Total cells: {total_cells}")
    print(f"Free cells: {free_cells} ({100*free_cells/total_cells:.1f}%)")
    print(f"Occupied cells: {occupied_cells} ({100*occupied_cells/total_cells:.1f}%)")
    print(f"Unknown cells: {unknown_cells} ({100*unknown_cells/total_cells:.1f}%)")
    
    print(f"\nMap generation complete!")
    print(f"Use in ROS with: <node name=\"map_server\" pkg=\"map_server\" type=\"map_server\" args=\"{args.output_yaml}\"/>")

if __name__ == "__main__":
    main()