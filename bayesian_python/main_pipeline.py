import numpy as np
import json  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from grid.grid_generator import GridGenerator
from utils.bim_integration import create_complete_integration
from utils.bayesian_repulsion_updater import BayesianRepulsionUpdater
from fields.repulsive_field import compute_repulsive_field
from pathfinding.mhastar import MultiHeuristicAStar

def load_bim_start_end_positions():
  
    with open("utils/extracted_objects.json", "r") as f:
        bim_data = json.load(f)
    
    # Get Agent position (start)
    agent_data = bim_data.get("Agent")
    if not agent_data:
        raise ValueError("Agent object not found in BIM data")
    start_world = np.array(agent_data["location"])
    
    # Get Capsule position (goal) 
    capsule_data = bim_data.get("Capsule")
    if not capsule_data:
        raise ValueError("Capsule object not found in BIM data")
    end_world = np.array(capsule_data["location"])
    
    print(f"   Navigation: from Agent to Capsule")
    print(f"   Start: {start_world}")
    print(f"   Goal:  {end_world}")
    print(f"   Direct distance: {np.linalg.norm(end_world - start_world):.2f}m")
    
    return start_world, end_world

def main():

    
    print("STARTING BAYESIAN PATHFINDING PIPELINE")
       
    # Generate Navigation Grid from BIM Mesh
    print("\nGenerating navigation grid from BIM mesh...")
    
    grid_gen = GridGenerator(
        mesh_file="bim/dicelab_bim.stl",
        cell_size=0.025,  # 25cm resolution for good performance
        ray_height=0.7
    )
    
    walkable_points, obstacle_points = grid_gen.generate_grid()
    grid = grid_gen.get_grid()
    
    print(f"   Generated {len(grid)} x {len(grid[0])} navigation grid")
    print(f"   Walkable nodes: {len(walkable_points)}")
    print(f"   Obstacle nodes: {len(obstacle_points)}")
    
    
    # Load BIM Integration and Objects
    
    print("\nLoading BIM integration...")
    
    integration = create_complete_integration('utils/extracted_objects.json')
    bayesian_obstacles = integration.get_obstacles_for_bayesian()
    repulsive_obstacles = integration.get_obstacles_for_repulsive_field()
    
    print(f"Loaded {len(bayesian_obstacles)} BIM obstacles")
    
    
    # Bayesian Environmental Reasoning
    
    print("\nPerforming Bayesian environmental analysis...")
    
    # Actual OpenAI API key
    
    API_KEY = "API_KEY"  
    
    try:
        bayesian_updater = BayesianRepulsionUpdater(
            api_key=API_KEY,
            obstacles=bayesian_obstacles
        )
        
        # Perform GPT communication and Bayesian updates
        bayesian_updater.perform_update()
        print("Bayesian environmental analysis complete")
        
        # Update repulsive obstacles with new gains
        integration.update_repulsive_gains()
        print("Obstacle gains updated from Bayesian results")
        
    except Exception as e:
        print(f"  Bayesian update failed (API key issue?): {e}")
        print("   Continuing with default danger levels...")
    
   
    # Compute Repulsive Potential Field
    
    print("\nComputing repulsive potential field...")
    
    compute_repulsive_field(grid, repulsive_obstacles)
    
    # Count nodes with potential
    nodes_with_potential = 0
    total_potential = 0
    walkable_count = 0
    
    for row in grid:
        for node in row:
            if node and node.walkable:
                walkable_count += 1
                if node.repulsive_potential > 0:
                    nodes_with_potential += 1
                    total_potential += node.repulsive_potential
    
    avg_potential = total_potential / walkable_count if walkable_count > 0 else 0
    print(f"   Repulsive field computed")
    print(f"   {nodes_with_potential}/{walkable_count} nodes affected")
    print(f"   Average potential: {avg_potential:.4f}")
    
    
    # Load Start and End Positions from BIM
    
    print("\nLoading navigation waypoints from BIM...")
    

    start_world, end_world = load_bim_start_end_positions()
    
    if start_world is None or end_world is None:
        print("Could not find valid start/end positions near BIM objects")
        return
    
    # Find closest walkable nodes
    start_node = grid_gen.closest_walkable_node(start_world)
    end_node = grid_gen.closest_walkable_node(end_world, allow_unwalkable=True)
    
    if start_node is None or end_node is None:
        print("    Could not find valid start/end nodes even with walkable positions")
        print(f"   Start node: {start_node}")
        print(f"   End node: {end_node}")
        return
    
    print(f"   Navigation nodes found:")
    print(f"   Start node: [{start_node.grid_x}, {start_node.grid_y}] walkable: {start_node.walkable}")
    print(f"   End node:   [{end_node.grid_x}, {end_node.grid_y}] walkable: {end_node.walkable}")
    
    # Multi-Heuristic A* Pathfinding
    
    print("\n  Running pathfinding...")

    #Verify potential field exists before pathfinding
    total_potential = 0
    potential_nodes = 0
    for row in grid:
        for node in row:
            if node and node.walkable and node.repulsive_potential > 0:
                potential_nodes += 1
                total_potential += node.repulsive_potential

    avg_potential = total_potential / potential_nodes if potential_nodes > 0 else 0
    print(f"   Pre-pathfinding verification:")
    print(f"   {potential_nodes} nodes with potential > 0")
    print(f"   Average potential: {avg_potential:.4f}")

    # Create pathfinder
    pathfinder = MultiHeuristicAStar(
        grid_generator=grid_gen,
        start_pos=start_world,
        goal_pos=end_world,
        w1=1.0,  # Heuristic inflation
        w2=2.0,  # Inadmissible bound
        gamma=5.0  # Potential field weight
    )

    pathfinder.grid = grid  # Use the grid that already has computed potentials!
    pathfinder.start_node = start_node  # Use the nodes we already found
    pathfinder.goal_node = end_node

    path_nodes = pathfinder.search()

    if path_nodes and len(path_nodes) > 0:
        actual_end_pos = path_nodes[-1].world_position
        print(f"Path found with {len(path_nodes)} nodes")
        
        # Show potential values along the path to verify they're being used
        print("Potential values along the path:")
        for i, node in enumerate(path_nodes[::max(1, len(path_nodes)//5)]):  # Sample 5 points
            print(f"   Path[{i}]: [{node.grid_x},{node.grid_y}] potential={node.repulsive_potential:.4f}")
        
        # Mark path nodes for visualization
        for node in path_nodes:
            node.is_path = True
            
        # Compute path metrics
        path_length = pathfinder.compute_path_length(path_nodes)
        min_dist, avg_dist = pathfinder.compute_distance_metrics(path_nodes, repulsive_obstacles)
        
        print(f"   Path Metrics:")
        print(f"   Total length: {path_length:.2f}m")
        print(f"   Min obstacle distance: {min_dist:.3f}m")
        print(f"   Avg obstacle distance: {avg_dist:.3f}m")
        
    else:
        print(" No path found")
        actual_end_pos = end_world
        path_nodes = []
   
    # Visualization

    print("\n Generating visualization...")
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection="3d")
    
    # Add BIM mesh
    verts = np.asarray(grid_gen.mesh.vertices)
    tris = verts[np.asarray(grid_gen.mesh.triangles)]
    mesh = Poly3DCollection(tris, alpha=0.2, linewidths=0.5, edgecolor="black")
    mesh.set_facecolor((0.8, 0.8, 0.8))
    ax.add_collection3d(mesh)
    
    # Prepare grid visualization
    half = grid_gen.node_diameter / 2.0
    z = grid_gen.base_z + 1e-3
    walkable_faces, obstacle_faces, path_faces, potential_faces = [], [], [], []
    
    for row in grid:
        for node in row or []:
            if not node:
                continue
                
            x, y, _ = node.world_position
            face = [
                [x - half, y - half, z],
                [x + half, y - half, z], 
                [x + half, y + half, z],
                [x - half, y + half, z]
            ]
            
            if getattr(node, "is_path", False):
                path_faces.append(face)
            elif not node.walkable:
                obstacle_faces.append(face)
            elif node.repulsive_potential > avg_potential * 2:  # High potential areas
                potential_faces.append(face)
            else:
                walkable_faces.append(face)
    
    # Add grid faces to plot
    if walkable_faces:
        ax.add_collection3d(Poly3DCollection(walkable_faces, facecolors="lightgreen", 
                                           edgecolors='none', alpha=0.6))
    if obstacle_faces:
        ax.add_collection3d(Poly3DCollection(obstacle_faces, facecolors="red", 
                                           edgecolors='none', alpha=0.8))
    if potential_faces:
        ax.add_collection3d(Poly3DCollection(potential_faces, facecolors="orange", 
                                           edgecolors='none', alpha=0.7))
    if path_faces:
        ax.add_collection3d(Poly3DCollection(path_faces, facecolors="blue", 
                                           edgecolors='none', alpha=1.0))
    
    # Add start and end markers
    cube_size = 0.2
    
    # Start marker (green cube)
    start_cube = create_cube_faces(start_world, cube_size)
    ax.add_collection3d(Poly3DCollection(start_cube, facecolors="green", alpha=0.8))
    
    # End marker (red cube)
    end_cube = create_cube_faces(actual_end_pos, cube_size)
    ax.add_collection3d(Poly3DCollection(end_cube, facecolors="red", alpha=0.8))
    
    # Legend
    ax.scatter([], [], [], c="lightgreen", label="walkable", alpha=0.6)
    ax.scatter([], [], [], c="red", label="obstacles", alpha=0.8)
    ax.scatter([], [], [], c="orange", label="high potential", alpha=0.7)
    ax.scatter([], [], [], c="blue", label="path", alpha=1.0)
    ax.scatter([], [], [], c="green", label="start")
    ax.scatter([], [], [], c="black", label="goal")
    
    # Set plot bounds
    min_b, max_b = verts.min(axis=0), verts.max(axis=0)
    margin = 0.1 * (max_b - min_b)
    ax.set_xlim(min_b[0] - margin[0], max_b[0] + margin[0])
    ax.set_ylim(min_b[1] - margin[1], max_b[1] + margin[1])
    ax.set_zlim(min_b[2] - margin[2], max_b[2] + margin[2])
    ax.set_box_aspect((max_b - min_b))
    
    # Camera angle (top-down view)
    ax.view_init(elev=90, azim=-90)
    ax.set_axis_off()
    ax.legend(loc="upper right")
    
    plt.title("Bayesian Multi-Heuristic A* Pathfinding", 
              fontsize=14, pad=20)
    
    print("   Visualization ready")
    print("\n BAYESIAN PATHFINDING PIPELINE COMPLETE!")
    print("   Close the plot window to exit.")
    
    plt.show(block=True)

def create_cube_faces(center, size):
    """Create faces for a cube marker at given center position"""
    half = size / 2
    x, y, z = center
    
    # Define 6 faces of a cube
    faces = [
        # Bottom face
        [[x-half, y-half, z-half], [x+half, y-half, z-half], 
         [x+half, y+half, z-half], [x-half, y+half, z-half]],
        # Top face  
        [[x-half, y-half, z+half], [x+half, y-half, z+half],
         [x+half, y+half, z+half], [x-half, y+half, z+half]],
        # Front face
        [[x-half, y-half, z-half], [x+half, y-half, z-half],
         [x+half, y-half, z+half], [x-half, y-half, z+half]],
        # Back face
        [[x-half, y+half, z-half], [x+half, y+half, z-half],
         [x+half, y+half, z+half], [x-half, y+half, z+half]],
        # Left face
        [[x-half, y-half, z-half], [x-half, y+half, z-half],
         [x-half, y+half, z+half], [x-half, y-half, z+half]],
        # Right face
        [[x+half, y-half, z-half], [x+half, y+half, z-half],
         [x+half, y+half, z+half], [x+half, y-half, z+half]]
    ]
    
    return faces

if __name__ == "__main__":
    main()