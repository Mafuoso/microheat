# microheat

**Simulating Microscopic Models of "Heat Rising"**

## Overview

This project aims to get fundamentally microscopic about a deceptively simple question: **Why does heat rise?**

While we have thermodynamic and continuum explanations (buoyancy, convection, etc.), there isn't a clear microscopic, particle-level explanation of this phenomenon. This simulation is designed to test a theory about the fundamental mechanisms behind heat rising using molecular dynamics.

## The Approach

### Molecular Dynamics (MD) Simulation
- **2D ideal gas model** with hard-sphere elastic collisions
- **Small particle numbers** (prioritizing accuracy over scale)
- **No interaction potentials** (just elastic collisions between particles)
- **Exact dynamics** - we're not approximating, we're simulating the real physics

### Why This Matters
Standard explanations of heat rising rely on macroscopic properties like density and pressure gradients. But what's actually happening at the particle level? How do individual molecular collisions lead to the emergent behavior we call "heat rising"? This simulation aims to answer that question.

## Current Implementation

### Physics Model
- **Particles**: Point masses with position (x, y), velocity (vx, vy), radius, and mass
- **Box**: Rectangular container with reflecting boundaries
- **Collisions**: Event-driven elastic collision detection and response
  - Particle-particle collisions
  - Particle-wall collisions
  - Conservation of momentum and energy
- **Temperature**: Defined through kinetic energy / velocity distributions
- **Gravity**: Ballistic motion with gravitational acceleration (g = 9.8)

### Velocity Initialization
Two methods for setting up initial conditions:

1. **`init_velocities_equiparition(particles, temperature, k_B=1.0)`**
   - Initializes all particles at the same temperature
   - Uses equipartition theorem: `KE = (3/2) k_B T`
   - Random velocity directions

2. **`init_hot_particle(particles, hot_index, hot_temperature, cold_temperature, k_B=1.0)`**
   - Sets one particle to high temperature ("hot")
   - All others at low temperature ("cold")
   - Allows studying energy transfer dynamics

### Visualization
- **Static plots**: Particles displayed as circles with accurate physical sizes, color-coded by speed
- **Velocity vectors**: Shown as arrows indicating direction and magnitude
- **Smooth animations**: Interpolated motion between collision events using ballistic trajectories
- **Event-driven simulation**: Efficient collision detection using priority queue
- **Progress tracking**: Visual feedback with tqdm progress bars during simulation

## Installation & Requirements

```bash
pip install numpy matplotlib tqdm
```

**Dependencies:**
- Python 3.x
- NumPy (array operations, math functions)
- Matplotlib (visualization and animation)
- tqdm (progress bars)

## Usage

### Quick Start - Run the Demo

```bash
python3 microheat.py
```

This generates animated simulations showing particle dynamics with collision physics.

### Using in Your Own Code

#### Static Visualization

```python
import microheat

# Initialize particles in a grid (uses spacing based on particle radius)
N = 16  # number of particles
particles, box = microheat.initialize(N, width=300.0, height=300.0)

# Set up velocities - Method 1: All same temperature
microheat.init_velocities_equiparition(particles, temperature=10, k_B=1.0)

# OR Method 2: One hot particle
microheat.init_hot_particle(particles, hot_index=0,
                           hot_temperature=50,
                           cold_temperature=5, k_B=1.0)

# Create static visualization
microheat.visualize_particles(particles, box,
                             title="My Simulation",
                             save_file="output.png")
```

#### Animated Simulation

```python
import microheat

# Initialize sparse configuration (ideal gas approximation)
particles, box = microheat.initialize(N=25, width=3000.0, height=3000.0)
microheat.init_velocities_equiparition(particles, temperature=10, k_B=1.0)

# Create smooth animation with interpolated motion
microheat.animate_simulation(particles, box,
                            max_time=20.0,  # simulation duration
                            fps=30,          # frames per second
                            save_file="simulation.gif",
                            title="Ideal Gas Simulation")
```

**Animation Features:**
- Smooth interpolation between collision events using ballistic trajectories
- Accurate particle sizes in visualization
- Progress bar shows simulation status
- Frame count automatically calculated based on physics timing

## Roadmap / Next Steps

### Completed ✓
- ✓ Event-driven collision detection between particles
- ✓ Particle-particle elastic collisions with momentum/energy conservation
- ✓ Particle-wall collisions
- ✓ Time evolution with ballistic trajectories
- ✓ Smooth animations with interpolated motion
- ✓ Accurate particle size visualization

### Current Priorities
1. **Add measurement tools**
   - Track kinetic energy distribution over time
   - Measure temperature gradients (spatial distribution)
   - Record collision statistics and rates
   - Monitor energy conservation

2. **Optimize performance**
   - Reduce event count for long simulations
   - Consider spatial hashing for collision detection
   - Optimize frame generation for animations

3. **Enhanced visualization**
   - Heat maps showing temperature distribution
   - Energy distribution histograms
   - Trajectory tracking for individual particles

### Future Extensions
- Temperature-controlled walls (thermal reservoirs)
- Variable particle masses and radii
- Statistical mechanics observables (pressure, entropy)
- 3D extension of the simulation
- Compare with theoretical predictions (Maxwell-Boltzmann distribution, etc.)

## Project Philosophy

This is about **getting it right**, not getting it fast. Small particle numbers with exact dynamics are preferred over approximate large-scale simulations. The goal is to understand the fundamental microscopic mechanism, not to simulate realistic systems (yet).

## Contributing

This is a research/exploration project. Questions, suggestions, and discussions about the physics are welcome!

## License

See LICENSE file for details. 
