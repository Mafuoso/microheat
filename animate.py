"""
Animation and visualization functions for microheat simulations.

This module handles all visualization tasks, keeping the main microheat.py
focused on physics computations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq
from tqdm import tqdm

# Import physics functions from microheat
from microheat import (
    Particle, Box, initialize, initialize_events,
    init_velocities_equiparition, advance_particles, predict_new_collisions
)


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
    marker_size = (8 * particle_radius * points_per_data_unit) ** 2

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
