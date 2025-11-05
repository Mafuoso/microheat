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
        self.collision_count: int = 0
        
    def collision():
        
    
class Box():

    def __init__(self, width, height):
        self.width: float = width
        self.height: float = height
        self.gravity_strength: float = 9.8 
        
    def make_grid(self, N: int):
        """Create a grid of N particles evenly spaced in the box."""
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
    """Initialize N particles in a box of given width and height."""
    particles = []
    box = Box(width, height)
    X, Y = box.make_grid(N)
    for i in range(N):
        particles[i]  = Particle(X[i], Y[i])
    return particles, box

def init_velocities_equiparition(particles:list[Particle] ,temperature: int, k_B:float = 1.0):
    """Initialize particle velocities according to the equipartition theorem."""
    speed = np.sqrt(3*k_B*temperature/particles[0].m) # root mean square speed, assuming all particles have equal maass
    for p in particles:
        theta = np.random.uniform(0, 2*np.pi)
        p.vx = speed * np.cos(theta) # choosing random x and y components
        p.vy = speed * np.sin(theta)

def init_hot_particle(particles:list[Particle] ,hot_index:int, hot_temperature: int, cold_temperature:int, k_B:float = 1.0):
    """Make a single particle hot and leave the others cold"""
    #Initialize cold particles
    init_velocities_equiparition(particles, cold_temperature, k_B)

    #Make one particle hot
    speed = np.sqrt(3*k_B*hot_temperature/particles[hot_index].m) # root mean square speed, assuming all particles have equal maass
    theta = np.random.uniform(0, 2*np.pi)
    particles[hot_index].vx = speed * np.cos(theta) # choosing random x and y components
    particles[hot_index].vy = speed * np.sin(theta)






    
    
