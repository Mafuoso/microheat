import numpy as np
import math
import matplotlib.pyplot as plt


class Particle():

    def __init__(self, x, y):
        self.x: float = x
        self.y: float = y
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.r: float = 1.0  # particle radius
        self.m: float = 1.0  # particle mass
        self.collision_count: int = 0
        self.g = 9.8  # gravitational acceleration

    def collision(self):
        pass

    def predict_state(self,dt:float):
        """Predict the trajectory of the particle after time dt under gravity g."""
        g = self.g
        new_x = self.x + self.vx * dt # Kinematic equation for horizontal motion
        new_y = self.y + self.vy * dt - 0.5 * g * dt**2 # Kinematic equation for vertical motion under gravity
        new_vy = self.vy - g * dt # Update vertical velocity due to gravity
        return new_x, new_y, self.vx, new_vy
    
    def time_to_wall(self,left_wall:float, right_wall:float, bottom_wall:float, top_wall:float):
        """Calculate time to collide with the walls of the box. Return the time to the first wall"""

        #time to left wall
        t_left = 

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
        particles.append(Particle(X[i], Y[i]))
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


def visualize_particles(particles: list[Particle], box: Box, title: str = "Particle Visualization", save_file: str = None):
    """Visualize particles and their velocity vectors."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Extract particle positions and velocities
    x_positions = [p.x for p in particles]
    y_positions = [p.y for p in particles]
    vx = [p.vx for p in particles]
    vy = [p.vy for p in particles]

    # Calculate speeds for color coding
    speeds = [np.sqrt(p.vx**2 + p.vy**2) for p in particles]

    # Plot particles with color based on speed
    scatter = ax.scatter(x_positions, y_positions, c=speeds, s=200, alpha=0.7,
                         cmap='hot', edgecolors='black', linewidth=1.5)

    # Plot velocity vectors
    ax.quiver(x_positions, y_positions, vx, vy, angles='xy', scale_units='xy',
              scale=0.5, color='blue', alpha=0.6, width=0.003)

    # Draw box boundaries
    ax.plot([0, box.width, box.width, 0, 0],
            [0, 0, box.height, box.height, 0],
            'k-', linewidth=2)

    # Set plot limits and labels
    ax.set_xlim(-box.width*0.1, box.width*1.1)
    ax.set_ylim(-box.height*0.1, box.height*1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Speed', fontsize=12)

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_file}")
    else:
        plt.show()

    return fig, ax









    
    
