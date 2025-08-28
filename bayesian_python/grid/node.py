class Node:

    def __init__(self, x, y, walkable, world_position=None, nav_mesh_position=None):

        self.x = x
        self.y = y
        self.walkable = walkable
        self.world_position = world_position if world_position is not None else (0,0,0)
        self.nav_mesh_position = nav_mesh_position if nav_mesh_position is not None else self.world_position
        self.neighbors = []

        self.g = float('inf')
        self.h = float('inf')
        self.parent = None

        self.object_name = None        
        self.danger_coefficient = 0.0
        self.repulsive = 0.0              
        self.dynamic_repulsive = 0.0
        self.attractive = 0.0

        self.is_path = False  #for visualization flag

    @property
    def f(self):
        
        return self.g + self.h

   
    @property
    def repulsive_potential(self):
        
        return self.repulsive

    @repulsive_potential.setter
    def repulsive_potential(self, value):
        self.repulsive = value

    @property
    def dynamic_repulsive_potential(self):
       
        return self.dynamic_repulsive

    @dynamic_repulsive_potential.setter
    def dynamic_repulsive_potential(self, value):
        self.dynamic_repulsive = value

    @property
    def attractive_potential(self):
       
        return self.attractive

    @attractive_potential.setter
    def attractive_potential(self, value):
        self.attractive = value

    @property
    def grid_x(self):
       
        return self.x

    @property
    def grid_y(self):
        
        return self.y
    def reset(self):

        self.g = float('inf')
        self.h = float('inf')
        self.parent = None

    def update_a_star_costs(self,g_cost, h_cost, parent=None):

        self.g = g_cost
        self.h = h_cost
        self.parent =parent

    def __repr__(self):

        return f"Node(x={self.x}, y={self.y}, walkable={self.walkable}, f={self.f:.2f}, g={self.g:.2f}, h={self.h:.2f})"

    def __eq__(self, other):

        return isinstance(other, Node) and self.x == other.x and self.y == other.y

    def __hash__(self):

        return hash((self.x, self.y))

    def __lt__(self, other):
       
        if not isinstance(other, Node):
            return NotImplemented
        # Compare by grid coordinates for consistent ordering
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y

    def __le__(self, other):
    
        if not isinstance(other, Node):
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other):

        if not isinstance(other, Node):
            return NotImplemented
        return not self <= other

    def __ge__(self, other):
       
        if not isinstance(other, Node):
            return NotImplemented
        return not self < other