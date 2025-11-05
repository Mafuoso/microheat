import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



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


def update_particles(particle_dict, box, dt=0.01):
    """Update particle positions with simple physics"""
    for key, particle in particle_dict.items():
        # Apply gravity
        particle.vy -= box.gravity_strength * dt

        # Update positions
        particle.x += particle.vx * dt
        particle.y += particle.vy * dt

        # Boundary collisions with damping
        if particle.x - particle.r < 0:
            particle.x = particle.r
            particle.vx = -particle.vx * 0.8  # Bounce with energy loss
        elif particle.x + particle.r > box.width:
            particle.x = box.width - particle.r
            particle.vx = -particle.vx * 0.8

        if particle.y - particle.r < 0:
            particle.y = particle.r
            particle.vy = -particle.vy * 0.8
        elif particle.y + particle.r > box.height:
            particle.y = box.height - particle.r
            particle.vy = -particle.vy * 0.8


def visualize(particle_dict, box, num_frames=200, dt=0.01, save_file=None):
    """Create an animation of particles moving in the box"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, box.width)
    ax.set_ylim(0, box.height)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Microheat Particle Simulation')

    # Create scatter plot for particles
    x_data = [p.x for p in particle_dict.values()]
    y_data = [p.y for p in particle_dict.values()]
    scatter = ax.scatter(x_data, y_data, s=100, c='red', alpha=0.6)

    # Draw box boundaries
    ax.plot([0, box.width, box.width, 0, 0], [0, 0, box.height, box.height, 0], 'k-', linewidth=2)

    def update(frame):
        # Update physics
        update_particles(particle_dict, box, dt)

        # Update plot
        x_data = [p.x for p in particle_dict.values()]
        y_data = [p.y for p in particle_dict.values()]
        scatter.set_offsets(np.c_[x_data, y_data])

        return scatter,

    anim = FuncAnimation(fig, update, frames=num_frames, interval=20, blit=True)

    if save_file:
        anim.save(save_file, writer='pillow', fps=30)
        print(f"Animation saved to {save_file}")
    else:
        plt.show()

    return anim


def run_demo(N=25, width=100.0, height=100.0, num_frames=200):
    """Run a simple demo of the particle system"""
    print(f"Initializing {N} particles in a {width}x{height} box...")
    particles, box = initialize(N, width, height)

    print("Starting animation...")
    visualize(particles, box, num_frames=num_frames)


if __name__ == "__main__":
    run_demo()