# microheat
Simulating Microscopic Models "Heat Rising"

## Features

- Initialize particles in an evenly-spaced grid within a box
- Simple physics simulation with gravity and boundary collisions
- Visualization and animation of particle motion

## Usage

### Quick Demo

Run the demo directly:
```bash
python3 microheat.py
```

### In Python Code

```python
import microheat

# Initialize particles
N = 25  # Number of particles
particles, box = microheat.initialize(N, width=100.0, height=100.0)

# Run visualization
microheat.visualize(particles, box, num_frames=200)

# Or save as GIF
microheat.visualize(particles, box, num_frames=200, save_file='animation.gif')
```

## Requirements

- numpy
- matplotlib
- pillow (for saving animations) 
