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
- **Collisions**: Will implement particle-particle elastic collisions (in progress)
- **Temperature**: Defined through kinetic energy / velocity distributions
- **No gravity** (yet) - testing pure kinetic effects first

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
- Particles displayed as circles, color-coded by speed
- Velocity vectors shown as arrows
- Real-time visualization of the microscopic dynamics

## Installation & Requirements

```bash
pip install numpy matplotlib
```

**Dependencies:**
- Python 3.x
- NumPy (array operations, math functions)
- Matplotlib (visualization)

## Usage

### Quick Start - Run the Demo

```bash
python3 microheat.py
```

This generates two visualization examples:
1. All particles at equilibrium temperature
2. One hot particle among cold particles

### Using in Your Own Code

```python
import microheat

# Initialize particles in a grid
N = 16  # number of particles
particles, box = microheat.initialize(N, width=100.0, height=100.0)

# Set up velocities - Method 1: All same temperature
microheat.init_velocities_equiparition(particles, temperature=10, k_B=1.0)

# OR Method 2: One hot particle
microheat.init_hot_particle(particles, hot_index=0,
                           hot_temperature=50,
                           cold_temperature=5, k_B=1.0)

# Visualize
microheat.visualize_particles(particles, box,
                             title="My Simulation",
                             save_file="output.png")
```

## Roadmap / Next Steps

### Immediate Priorities
1. **Implement collision detection** between particles
   - Particle-particle elastic collisions
   - Conservation of momentum and energy

2. **Time evolution**
   - Update particle positions based on velocities
   - Detect and handle collisions at each timestep
   - Create animations of evolving system

3. **Add measurement tools**
   - Track kinetic energy distribution
   - Measure temperature gradients
   - Record collision statistics

### Future Extensions
- Optional gravity field
- Temperature-controlled walls (thermal reservoirs)
- Measurement of spatial temperature profiles
- Statistical mechanics observables (pressure, temperature, entropy)
- Compare with theoretical predictions

## Project Philosophy

This is about **getting it right**, not getting it fast. Small particle numbers with exact dynamics are preferred over approximate large-scale simulations. The goal is to understand the fundamental microscopic mechanism, not to simulate realistic systems (yet).

## Contributing

This is a research/exploration project. Questions, suggestions, and discussions about the physics are welcome!

## License

See LICENSE file for details. 
