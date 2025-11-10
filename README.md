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

## The Theory

This project tests my theory that heat rises through a **microscopic mechanism** driven by a collision bias, not relying on density or buoyoancy to explain the process. 

### The Proposed Mechanism

**Starting Condition:**
- Ideal gas with gravity and elastic collisions between hard spheres
- Can start uniform or with temperature variations
- Can start at thermal equillibrium 

**Step 1: Ballistic Stratification**
- All particles undergo parabolic motion under gravity
- Hot (high energy) particles: larger velocities → reach greater heights (h_max ~ v²/2g) → longer flight times **ON AVERAGE**
- Cold (low energy) particles: smaller velocities → gravity pulls them down faster/ less Vy to oppose g **ON AVERAGE**

**Step 2: Geometric Collision Bias**
- Hot particle spends more time aloft → when it descends, statistically more likely to encounter cold particle *beneath* it
- Not because cold particles don't move up, but because the time-averaged probability distributions are stratified
- Collision geometry is biased: hot particle colliding downward into cold particle below is a more likely event 

**Step 3: Upward Redirection**
- Downward-moving hot particle hits slower cold particle below
- Elastic collision can redirect hot particle upward
- Hot particle reaches higher than pure ballistics would allow from its initial energy

**Step 4: Self-Reinforcing Cycle**
- Enhanced elevation → more time aloft → more biased collision geometry → maintains/increases elevation
- This creates a positive feedback loop
- Doesn't mean that hot air will be at the top of the box, but it does mean it is more likely to find the hot particle near the top or above the cold particles **ON AVERAGE**

### Key Claims

1. **Single-Particle Phenomenon:** Even one hot particle in a cold bath should stay preferentially elevated because the geometric bias from ballistic stratification is sufficient to overcome thermalization.

2. **No Macroscopic Prerequisites:** The mechanism doesn't require:
   - Density differences
   - Pressure gradients
   - Large ensembles
   - Statistical mechanics ensemble averaging

3. **Microscopic Foundation:** With many hot particles, each independently experiences this effect, so we observe systematic upward transport of thermal energy. The mechanism is fundamentally microscopic.

### What Makes It Work

The collision rate and geometry are not uniform in space - they're correlated with the energy-height stratification that gravity + ballistics naturally creates. The bias is probabilistic but persistent.

### Testing the Theory

- Work In Progress

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

## Project Structure

- **`microheat.py`** - Core physics engine with `simulate()` function
  - Particle and Box classes
  - Collision detection and event queue
  - `simulate()` - Main simulation function with configurable parameters
- **`animate.py`** - Visualization and animation functions
- **`experiments.py`** - Independent experiment configurations using `simulate()`

## Usage

### Quick Start - Run All Experiments

```bash
python3 experiments.py
```

This runs all configured experiments, generating:
- Animated visualizations (GIF files)
- Height time series plots (PNG files)
- Temperature-height correlation analysis
- Console output with statistics

Each experiment is an independent function that can be run separately.

### Run Individual Experiments

```python
from experiments import experiment_equipartition, experiment_hot_particle, experiment_temp_height_correlation

# Experiment 1: Equipartition baseline
experiment_equipartition(N=50, width=3000.0, height=3000.0,
                        temperature=10.0, max_time=50.0, fps=10)

# Experiment 2: One hot particle
experiment_hot_particle(N=50, width=3000.0, height=3000.0,
                       hot_index=5, hot_temperature=500.0,
                       cold_temperature=10.0, max_time=100.0, fps=30)

# Experiment 3: Temperature-height correlation
results = experiment_temp_height_correlation(
    temp_list=[50, 100, 200, 300, 400, 500],
    hot_index=50,
    ntrials=10
)
```

### Using the simulate() Function

The core `simulate()` function in `microheat.py` provides a flexible API for running simulations:

```python
from microheat import simulate

# Run simulation with hot particle and height tracking
particles, box, tracked_heights = simulate(
    hot_index=5,              # Index of hot particle (None for equipartition)
    hot_temperature=500,      # Temperature of hot particle
    cold_temperature=50,      # Temperature of other particles
    max_time=100,             # Simulation duration
    N=100,                    # Number of particles
    width=1000,               # Box width
    height=1000,              # Box height
    sample_interval=10,       # Height sampling interval
    track_indices=[5, 6, 7],  # Particles to track
    show_progress=True,       # Display progress bar
    k_B=1.0                   # Boltzmann constant
)

# tracked_heights is a dict: {particle_index: [heights over time]}
print(f"Mean height of particle 5: {sum(tracked_heights[5])/len(tracked_heights[5])}")
```

### Creating Custom Visualizations

```python
from microheat import simulate, initialize, init_hot_particle
from animate import animate_simulation, visualize_particles

# Method 1: Use simulate() for analysis, then create separate animation
particles_sim, box_sim, heights = simulate(hot_index=5, hot_temperature=500, max_time=100)

# Create animation separately (doesn't slow down simulation)
particles_anim, box_anim = initialize(N=100, width=1000, height=1000)
init_hot_particle(particles_anim, hot_index=5, hot_temperature=500, cold_temperature=50)
animate_simulation(particles_anim, box_anim, max_time=100, fps=30,
                  save_file="my_animation.gif", hot_particle_index=5)

# Method 2: Static visualization of final state
visualize_particles(particles_sim, box_sim,
                   title="Final State After 100 Time Units",
                   save_file="final_state.png",
                   hot_particle_index=5)
```

## Roadmap / Next Steps

### Completed ✓
- ✓ Event-driven collision detection between particles
- ✓ Particle-particle elastic collisions with momentum/energy conservation
- ✓ Particle-wall collisions
- ✓ Time evolution with ballistic trajectories
- ✓ Smooth animations with interpolated motion
- ✓ Accurate particle size visualization
- ✓ Height time series tracking and plotting
- ✓ Independent experiment configurations
- ✓ `simulate()` function API for flexible simulations
- ✓ Multi-trial temperature-height correlation analysis
- ✓ Parallel processing for multiple simulation runs

### Current Priorities
1. **Additional measurement tools**
   - Track kinetic energy distribution over time
   - Measure temperature gradients (spatial distribution)
   - Record collision statistics and rates
   - Monitor energy conservation
2. **Enhanced analysis**
   - Velocity distribution histograms
   - Energy conservation validation plots 

## Project Philosophy

This is about **getting it right**, not getting it fast. Small particle numbers with exact dynamics are preferred over approximate large-scale simulations. The goal is to understand the fundamental microscopic mechanism, not to simulate realistic systems. Ideally (hahah) we will be able to verify the theory of heat rising based on particle collisions in ideal gasses. Though I am still designing the experiment to verify the theory and perhaps I will elucidate on what I think is happening here as well. 

## Contributing

This is a research/exploration project. Questions, suggestions, and discussions about the physics are welcome!
Email Me: mu2faroo@uwaterloo.ca

## License

See LICENSE file for details. 
