import numpy as np
from heapq import heappush, heappop
from collections import deque
from pathfinding_simulation.grid.grid_generator_open3d import GridGenerator
from pathfinding_simulation.fields.repulsive_field import RepulsiveObstacle
from pathfinding_simulation.utils.bayesian_repulsion_updater import BayesianRepulsionUpdater
from pathfinding_simulation.grid.node import Node
import rospy
class MultiHeuristicAStar:
    def __init__(self, grid_generator, start_pos, goal_pos, w1=1.0, w2=2.0, gamma=5.0):
        self.grid_gen = grid_generator
        self.grid = self.grid_gen.get_grid()
        self.start_node = self.grid_gen.closest_walkable_node(start_pos)
        self.goal_node = self.grid_gen.closest_walkable_node(goal_pos, allow_unwalkable=True)
        self.w1 = w1
        self.w2 = w2
        self.gamma = gamma

        self.open0 = []  # Anchor queue (admissible)
        self.open1 = []  # Inadmissible queue
        self.closed_anchor = set()
        self.closed_inad = set()

    def compute_path_length(self, path):
        length = 0.0
        for i in range(1, len(path)):
            a = np.array(path[i - 1].world_position)
            b = np.array(path[i].world_position)
            length += np.linalg.norm(a - b)
        return length

    def compute_distance_metrics(self, path, obstacles):

        if not path:
            return (0.0, 0.0)

        sum_dist = 0.0
        min_dist = float('inf')

        for node in path:
            wp = np.array(node.world_position)
            closest_dist = float('inf')

            for obstacle in obstacles:
                pt = np.array(obstacle.get_closest_point(wp))
                dist = np.linalg.norm(wp - pt)
                if dist < closest_dist:
                    closest_dist = dist

            sum_dist += closest_dist
            if closest_dist < min_dist:
                min_dist = closest_dist

        avg_dist = sum_dist / len(path)
        return (min_dist, avg_dist)

    def plan_path(self, start_position, goal_position, bayes_manager=None, repulsive_field=None, obstacles=None, visualize_fn=None):
        if start_position is None or goal_position is None:
            rospy.loginfo("Error: Please assign start and goal positions.")
            return

        if bayes_manager is not None:
            bayes_manager.perform_update()
        else:
            rospy.loginfo("Warning: No RepulsiveBayesManager found; skipping Bayesian step.")

        self.grid = self.grid_gen.get_grid()

        if repulsive_field is not None:
            repulsive_field.compute_repulsive_field()

        total_nodes = len(self.grid) * len(self.grid[0])
        rospy.loginfo(f"[Grid] Total nodes: {total_nodes}")

        self.start_node = self.grid_gen.closest_walkable_node(start_position)
        self.goal_node = self.grid_gen.closest_walkable_node(goal_position, allow_unwalkable=True)
        rospy.loginfo(f"[Grid] Total nodes: {len(self.grid_gen.get_all_nodes())}")
        rospy.loginfo(f"Start grid[{self.start_node.grid_x},{self.start_node.grid_y}] walkable? {self.start_node.walkable}")
        rospy.loginfo(f"Goal grid[{self.goal_node.grid_x},{self.goal_node.grid_y}] walkable? {self.goal_node.walkable}")

        path = self.search()
        rospy.loginfo(f"Path nodes: {len(path)}")

        if visualize_fn is not None:
            visualize_fn(path)

        total_world_length = self.compute_path_length(path)
        min_dist, avg_dist = self.compute_distance_metrics(path, obstacles) if obstacles else (0.0, 0.0)

        rospy.loginfo(f" Path length = {total_world_length:.2f} units, "
            f"Min-obstacle distance = {min_dist:.3f} m, "
            f"Avg-obstacle distance = {avg_dist:.3f} m")


    def h0(self, node):
        return np.linalg.norm(np.array(node.world_position) - np.array(self.goal_node.world_position))

    def h1(self, node):
        return node.repulsive_potential


    def key(self, node, i):
        hi = self.h0(node) if i == 0 else self.h1(node)
        return node.g + self.w1 * hi

    def search(self):
        # Initialization 
        self.open0 = []
        self.open1 = []
        self.closed_anchor.clear()
        self.closed_inad.clear()

        # Reset all nodes
        width = len(self.grid)
        height = len(self.grid[0])
        for x in range(width):
            for y in range(height):
                node = self.grid[x][y]
                node.g = float('inf')
                node.parent = None

        self.start_node.g = 0.0
        heappush(self.open0, (self.key(self.start_node, 0), self.start_node))
        heappush(self.open1, (self.key(self.start_node, 1), self.start_node))

        # Main loop 
        while self.open0 and self.key(self.open0[0][1], 0) < float('inf'):
            # Decide which queue to expand
            use_inad = (self.open1 and
                    self.key(self.open1[0][1], 1) <= self.w2 * self.key(self.open0[0][1], 0))

            if use_inad:
                # Termination check for inadmissible
                if self.goal_node.g <= self.key(self.open1[0][1], 1):
                    break
                _, node = heappop(self.open1)
                self.expand_state(node, from_inadmissible=True)
            else:
                # Termination check for anchor
                if self.goal_node.g <= self.key(self.open0[0][1], 0):
                    break
                _, node = heappop(self.open0)
                self.expand_state(node, from_inadmissible=False)

        return self.reconstruct_path(self.goal_node)

    def expand_state(self, node, from_inadmissible):
        # Remove from both open queues 
        self.open0 = [(k, n) for (k, n) in self.open0 if n != node]
        self.open1 = [(k, n) for (k, n) in self.open1 if n != node]
        # Re-heapify after removals
        import heapq
        heapq.heapify(self.open0)
        heapq.heapify(self.open1)

        if from_inadmissible:
            self.closed_inad.add(node)
        else:
            self.closed_anchor.add(node)

        for neighbor in self.grid_gen.get_neighbors(node):
            if not neighbor.walkable:
                continue

            d = np.linalg.norm(np.array(node.world_position) - np.array(neighbor.world_position))
            pot_cost = self.gamma * neighbor.repulsive_potential
            tentative_g = node.g + d + pot_cost

            if tentative_g < neighbor.g:
                neighbor.g = tentative_g
                neighbor.parent = node

                if neighbor not in self.closed_anchor:
                    heappush(self.open0, (self.key(neighbor, 0), neighbor))

                if (neighbor not in self.closed_inad and
                        self.key(neighbor, 1) <= self.w2 * self.key(neighbor, 0)):
                    heappush(self.open1, (self.key(neighbor, 1), neighbor))
                    
    def reconstruct_path(self, end_node):
        path = []
        if end_node.parent is None:
            return path

        current = end_node
        while current is not None:
            path.append(current)
            current = current.parent

        path.reverse()
        return path

