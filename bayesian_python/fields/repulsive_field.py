import numpy as np
from typing import List, Tuple

class RepulsiveObstacle:
    def __init__(self, get_closest_point_fn, gain=1.0, max_influence=0.5):

        self.get_closest_point = get_closest_point_fn
        self.gain = gain
        self.max_influence = max_influence


def compute_repulsive_field(grid, obstacles: List[RepulsiveObstacle], default_gain=1.0, default_max_dist=0.5):

    width, height = len(grid), len(grid[0])
    raw = np.zeros((width, height), dtype=np.float32)

    for x in range(width):
        for y in range(height):
            node = grid[x][y]
            if not node.walkable:
                raw[x, y] = 0.0
                continue

            wp = np.array(node.world_position)
            total_repulsion = 0.0

            for obs in obstacles:
                closest_point = np.array(obs.get_closest_point(wp))
                d = np.linalg.norm(wp - closest_point)

                if d >= obs.max_influence:
                    continue

                strength = np.exp(-d)
                total_repulsion += obs.gain * strength

            raw[x, y] = total_repulsion

    # Write results into nodes
    for x in range(width):
        for y in range(height):
            grid[x][y].repulsive_potential = raw[x, y]
