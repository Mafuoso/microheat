import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq
from tqdm import tqdm


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


    def predict_state(self,dt:float):
        """Predict the trajectory of the particle after time dt under gravity g."""
        g = self.g
        new_x = self.x + self.vx * dt # Kinematic equation for horizontal motion
        new_y = self.y + self.vy * dt - 0.5 * g * dt**2 # Kinematic equation for vertical motion under gravity
        new_vy = self.vy - g * dt # Update vertical velocity due to gravity
        return new_x, new_y, self.vx, new_vy
    
    def time_to_wall(self, right_wall:float, top_wall:float):
        """Calculate time to collide with the walls of the box. Return the time to the first wall"""
        #Check velocity direction
        if self.vx > 0:
            time_to_right = ((right_wall - self.r) - self.x) / self.vx
            time_to_left = float('inf')
        elif self.vx < 0:
            time_to_left = (self.r - self.x) / self.vx 
            time_to_right = float('inf')
        else:
            time_to_right = float('inf')
            time_to_left = float('inf')
                    
        #Top Wall
        top_equation = [-0.5*self.g, self.vy, self.y- top_wall + self.r,] # coefficients of the quadratic equation   
        top_roots = np.roots(top_equation)
        top_roots = top_roots[np.isreal(top_roots) & (top_roots >= 1e-10)].real # keep only real and positive roots
        time_to_top = min(top_roots) if len(top_roots) > 0 else float('inf')
        
        #Bottom Wall 
        bottom_equation = [-0.5*self.g, self.vy, self.y - self.r,]  # coefficients of the quadratic equation
        bottom_roots = np.roots(bottom_equation)
        bottom_roots = bottom_roots[np.isreal(bottom_roots) & (bottom_roots >= 1e-10)].real # keep only real and positive roots
        time_to_bottom = min(bottom_roots) if len(bottom_roots) > 0 else float('inf')
        
        #Return time to collision and which wall we are colliding with
        times = [time_to_left, time_to_right, time_to_bottom, time_to_top]
        min_time = min(times)
        if times.index(min_time) == 0:
            wall = 'left'
        elif times.index(min_time) == 1:
            wall = 'right'
        elif times.index(min_time) == 2:
            wall = 'bottom'
        else:
            wall = 'top'
            
        return min_time, wall
    
    def time_to_particle(self, particle):
        deltax = particle.x - self.x
        deltay = particle.y - self.y
        delta_vx = particle.vx - self.vx
        delta_vy = particle.vy - self.vy
        
        coeff_array = [delta_vx**2 + delta_vy**2,
                       2*(deltax*delta_vx + deltay*delta_vy),
                       deltax**2 + deltay**2 - (self.r + particle.r)**2]
        times = np.roots(coeff_array)
        times = times[np.isreal(times) & (times >= 1e-10)].real # keep only real and positive roots
        if len(times) == 0:
            return float('inf')
        else:
            return min(times)
        
    def collide_with_wall(self,wall:str):
        """Update velocity after collision with wall."""
        if wall == 'left' or wall == 'right':
            self.vx = -self.vx
        elif wall == 'top' or wall == 'bottom':
            self.vy = -self.vy
        self.collision_count += 1
        #TODO: Glacning collisions with side wall should send particle back up if it was falling down. Need to handle this differently
    
    def collide_with_particle(self, particle):
        """Update velocities after collision with another particle."""
        x1 = (self.x,self.y)
        x2 = (particle.x, particle.y)
        v1 = (self.vx, self.vy)
        v2 = (particle.vx, particle.vy)
        delta_v = np.array(v2) - np.array(v1)
        dx = np.array(x2) - np.array(x1)
        n_hat = dx / np.linalg.norm(dx)
        v1prime = np.array(v1) + np.dot(delta_v,n_hat)*n_hat # equal mass collision hence why the velocity update is symmetric
        v2prime = np.array(v2) - np.dot(delta_v,n_hat)*n_hat
        self.vx, self.vy = v1prime
        particle.vx, particle.vy = v2prime
        self.collision_count += 1
        particle.collision_count += 1
        
                
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

def initialize_events(particles:list[Particle],box:Box):
    """Initialize the event priority queue with wall and particle collision events."""
    events = []
    
    # Wall collision events
    for i, p in enumerate(particles):
        time_to_wall, wall = p.time_to_wall(box.width, box.height)
        count_i = p.collision_count
        if time_to_wall < float('inf'):
            heapq.heappush(events, (time_to_wall,i, wall, count_i, 0)) # (time, particle index, wall, count_i, count_j=0 for wall collisions)
    # Particle collision events
    
    for i,p in enumerate(particles):
        for j in range(i+1, len(particles)):
            q = particles[j]
            time_to_particle = p.time_to_particle(q)
            count_i = p.collision_count
            count_j = q.collision_count
            if time_to_particle < float('inf'):
                heapq.heappush(events, (time_to_particle, i, j, count_i, count_j)) # (time, particle i index, particle j index, count_i, count_j)
    return events
    

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


def advance_particles(particles: list[Particle], dt: float):
    """advance all particles by time dt."""
    for p in particles:
        new_x, new_y, new_vx, new_vy = p.predict_state(dt)
        p.x = new_x
        p.y = new_y
        p.vx = new_vx
        p.vy = new_vy
        
        
def predict_new_collisions(particles,i,box, events,current_time):
    """Predict new collisions after an event has occurred."""
    p = particles[i]
    # Wall collision
    time_to_wall, wall = p.time_to_wall(box.width, box.height)
    count_i = p.collision_count
    if time_to_wall < float('inf'):
        heapq.heappush(events, (time_to_wall + current_time, i, wall, count_i, 0))  # (time, particle index, wall, count_i, count_j=0 for wall collisions)

    # Particle collisions
    for j, q in enumerate(particles):
        if j != i:
            time_to_particle = p.time_to_particle(q)
            count_j = q.collision_count
            if time_to_particle < float('inf'):
                heapq.heappush(events, (time_to_particle + current_time, i, j, count_i, count_j))  # (time, particle i index, particle j index, count_i, count_j)
                
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

    # Calculate marker size based on actual particle radius
    # Convert radius from data units to points for accurate representation
    fig_width_inches = 10  # from figsize=(10, 10)
    data_width = box.width * 1.2  # plot shows box + 10% padding each side
    points_per_data_unit = (fig_width_inches * 72) / data_width  # 72 points per inch
    particle_radius = particles[0].r if particles else 1.0
    marker_size = (2 * particle_radius * points_per_data_unit) ** 2

    # Plot particles with color based on speed
    scatter = ax.scatter(x_positions, y_positions, c=speeds, s=marker_size, alpha=0.7,
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


def create_smooth_frames(snapshot1, snapshot2, num_frames):
    """Interpolate particle positions between two collision events using ballistic motion."""
    frames = []
    dt_total = snapshot2['time'] - snapshot1['time']
    g = 9.8  # gravitational acceleration

    for frame_num in range(num_frames):
        t_frac = frame_num / num_frames
        dt = t_frac * dt_total

        # Predict each particle's position using ballistic trajectory
        frame = {
            'x': [], 'y': [], 'vx': [], 'vy': [],
            'time': snapshot1['time'] + dt,
            'event_type': 'In flight',
            'event_num': snapshot1['event_num']
        }

        for i in range(len(snapshot1['x'])):
            # Ballistic trajectory from snapshot1
            x = snapshot1['x'][i] + snapshot1['vx'][i] * dt
            y = snapshot1['y'][i] + snapshot1['vy'][i] * dt - 0.5 * g * dt**2
            vx = snapshot1['vx'][i]
            vy = snapshot1['vy'][i] - g * dt

            frame['x'].append(x)
            frame['y'].append(y)
            frame['vx'].append(vx)
            frame['vy'].append(vy)

        frames.append(frame)

    return frames


def animate_simulation(particles: list[Particle], box: Box, max_time: float = 10.0,
                       fps: int = 30, save_file: str = None, title: str = "Ideal Gas Simulation"):
    """
    Animate the particle simulation with smooth interpolation between collision events.

    Args:
        particles: List of Particle objects
        box: Box object containing the simulation boundaries
        max_time: Total simulation time
        fps: Frames per second for the animation
        save_file: If provided, save animation to this file (e.g., 'sim.gif' or 'sim.mp4')
        title: Title for the animation
    """
    # Initialize events
    events = initialize_events(particles, box)
    current_time = 0.0

    # Storage for event snapshots
    event_snapshots = []

    # Count total events for progress bar
    total_events = len(events)

    # Capture initial state
    initial_snapshot = {
        'x': [p.x for p in particles],
        'y': [p.y for p in particles],
        'vx': [p.vx for p in particles],
        'vy': [p.vy for p in particles],
        'time': 0.0,
        'event_type': 'Initial state',
        'event_num': 0
    }
    event_snapshots.append(initial_snapshot)

    # Run simulation and capture each event
    print(f"Running simulation for {max_time:.2f} time units...")
    pbar = tqdm(total=total_events, desc="Processing events", unit="event")

    event_count = 1
    while current_time < max_time and len(events) > 0:
        # Get next event
        (event_time, i, j, count_i, count_j) = heapq.heappop(events)

        # Validity check on the event
        if particles[i].collision_count != count_i:
            continue
        if isinstance(j, int) and particles[j].collision_count != count_j:
            continue

        # Don't process events beyond max_time
        if event_time > max_time:
            break

        # Advance all particles to event time
        advance_particles(particles, event_time - current_time)
        current_time = event_time

        # Determine event type for display
        if isinstance(j, int):  # Particle-Particle collision
            event_type = f"Particle {i} ↔ Particle {j}"
            particles[i].collide_with_particle(particles[j])
            predict_new_collisions(particles, i, box, events, current_time)
            predict_new_collisions(particles, j, box, events, current_time)
        else:  # Particle-Wall collision
            event_type = f"Particle {i} → {j.capitalize()} wall"
            particles[i].collide_with_wall(j)
            predict_new_collisions(particles, i, box, events, current_time)

        # Capture snapshot after collision
        snapshot = {
            'x': [p.x for p in particles],
            'y': [p.y for p in particles],
            'vx': [p.vx for p in particles],
            'vy': [p.vy for p in particles],
            'time': current_time,
            'event_type': event_type,
            'event_num': event_count
        }
        event_snapshots.append(snapshot)
        event_count += 1
        pbar.update(1)

    pbar.close()
    print(f"Simulation complete. Captured {len(event_snapshots)} collision events.")

    # Handle case of no events
    if len(event_snapshots) <= 1:
        print("Warning: No collision events occurred during the simulation.")
        print("Try using: smaller box, more particles, higher temperature, or longer max_time.")
        plt.close('all')
        return None

    # Create smooth interpolated frames between events
    print("Creating smooth interpolated frames...")
    frames_data = []
    for i in range(len(event_snapshots) - 1):
        snapshot1 = event_snapshots[i]
        snapshot2 = event_snapshots[i + 1]

        # Calculate number of frames based on time difference
        dt = snapshot2['time'] - snapshot1['time']
        num_frames = max(1, int(dt * fps))

        # Interpolate between snapshots
        interpolated_frames = create_smooth_frames(snapshot1, snapshot2, num_frames)
        frames_data.extend(interpolated_frames)

    # Add the final snapshot
    frames_data.append(event_snapshots[-1])

    print(f"Created animation with {len(frames_data)} total frames ({len(event_snapshots)} collision events)")

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate speed limits for consistent colorbar
    all_speeds = []
    for snapshot in event_snapshots:
        speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in zip(snapshot['vx'], snapshot['vy'])]
        all_speeds.extend(speeds)
    vmin, vmax = 0, max(all_speeds) if all_speeds else 1

    # Calculate marker size based on actual particle radius
    # Convert radius from data units to points for accurate representation
    fig_width_inches = 10  # from figsize=(10, 10)
    data_width = box.width * 1.2  # plot shows box + 10% padding each side
    points_per_data_unit = (fig_width_inches * 72) / data_width  # 72 points per inch
    particle_radius = particles[0].r if particles else 1.0
    marker_size = (8 * particle_radius * points_per_data_unit) ** 2

    # Initialize plot elements
    scatter = ax.scatter([], [], s=marker_size, alpha=0.7, cmap='hot',
                        edgecolors='black', linewidth=1.5, vmin=vmin, vmax=vmax, c=[])
    quiver_artists = []  # Store quiver objects
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       verticalalignment='top', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    event_text = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                        verticalalignment='bottom', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Draw box boundaries
    ax.plot([0, box.width, box.width, 0, 0],
            [0, 0, box.height, box.height, 0],
            'k-', linewidth=2)

    # Set plot properties
    ax.set_xlim(-box.width*0.1, box.width*1.1)
    ax.set_ylim(-box.height*0.1, box.height*1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Speed', fontsize=12)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_array(np.array([]))
        time_text.set_text('')
        event_text.set_text('')
        return [scatter, time_text, event_text]

    def update(frame_idx):
        nonlocal quiver_artists
        frame = frames_data[frame_idx]

        # Update particle positions
        positions = np.c_[frame['x'], frame['y']]
        scatter.set_offsets(positions)

        # Update colors based on speed
        speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in zip(frame['vx'], frame['vy'])]
        scatter.set_array(np.array(speeds))

        # Remove old quiver arrows
        for artist in quiver_artists:
            artist.remove()
        quiver_artists.clear()

        # Add new velocity vectors
        if len(frame['x']) > 0:
            quiver_new = ax.quiver(frame['x'], frame['y'], frame['vx'], frame['vy'],
                                  angles='xy', scale_units='xy', scale=0.5,
                                  color='blue', alpha=0.6, width=0.003)
            quiver_artists.append(quiver_new)

        # Update text displays
        time_text.set_text(f'Time: {frame["time"]:.2f}')
        event_text.set_text(f'Event #{frame["event_num"]}: {frame["event_type"]}')

        return [scatter, time_text, event_text] + quiver_artists

    anim = FuncAnimation(fig, update, init_func=init, frames=len(frames_data),
                        interval=1000/fps, blit=False, repeat=True)

    if save_file:
        print(f"Saving animation to {save_file}...")
        if save_file.endswith('.gif'):
            anim.save(save_file, writer='pillow', fps=fps)
        else:
            anim.save(save_file, writer='ffmpeg', fps=fps)
        print(f"Animation saved to {save_file}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    return anim


def run_demo():
    """
    Run demonstration animations of the ideal gas simulation.

    Shows smooth animation with interpolated motion between collision events.
    Uses sparse particle configurations appropriate for ideal gas behavior.
    """
    print("=== Microheat Smooth Animation Demo ===\n")
    print("Note: Animation uses smooth interpolation between collision events")
    print("      Using sparse configurations appropriate for ideal gas\n")

    # Demo 1: All particles at same temperature - SPARSE configuration
    print("Demo 1: Equipartition - All particles at temperature T=10")
    print("        (25 particles in 3000x3000 box)\n")
    particles1, box1 = initialize(N=25, width=3000.0, height=3000.0)
    init_velocities_equiparition(particles1, temperature=10, k_B=1.0)

    animate_simulation(particles1, box1, max_time=50.0, fps=10,
                      save_file="demo1_equipartition_animation.gif",
                      title="Ideal Gas: Equipartition at T=10")

    #Demo 2: One hot particle - SPARSE configuration
    print("Demo 2: One Hot Particle - Hot particle at T=100, others at T=10")
    print("        (25 particles in 3000x3000 box)\n")
    particles2, box2 = initialize(N=25, width=3000.0, height=3000.0)
    init_hot_particle(particles2, hot_index=5, hot_temperature=500,
                      cold_temperature=10, k_B=1.0)
    animate_simulation(particles2, box2, max_time=100.0, fps=10,
                      save_file="demo2_one_hot_particle_animation.gif",
                      title="Ideal Gas: One Hot Particle at T=100")
    

    print()

   
if __name__ == "__main__":
    run_demo()
