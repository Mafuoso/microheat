import numpy as np
import math



class Particle():

    def __init__(self, x, y):
        self.x: float = x
        self.y: float = y
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.r: float = 1.0  # particle radius
        self.m: float = 1.0  # particle mass
    
class Box():

    def __init__(self, width, height):
        self.width: float = width
        self.height: float = height
        self.gravity_strength: float = 9.8 
        
    def make_grid(self, N: int):
        # Calculate grid dimensions - make it roughly square
        n_cols = int(math.ceil(math.sqrt(N)))
        n_rows = int(math.ceil(N / n_cols))

        # Create evenly spaced arrays
        x_array = np.linspace(0, self.width, n_cols + 2)[1:-1]  # Exclude boundaries
        y_array = np.linspace(0, self.height, n_rows + 2)[1:-1]  # Exclude boundaries

        # Create meshgrid
        X, Y = np.meshgrid(x_array, y_array)

        # Flatten and return only N particles (in case n_rows * n_cols > N)
        X_flat = X.flatten()[:N]
        Y_flat = Y.flatten()[:N]

        return X_flat, Y_flat 
    
def initialize(N: int, width: float, height: float):
    particle_dict = {}
    box = Box(width, height)
    X, Y = box.make_grid(N)
    for i in range(N):
        particle_dict["particle{0}".format(i)] = Particle(X[i], Y[i])
    return particle_dict, box