import numpy as np
import math
import heapq


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
        top_roots = top_roots[np.isreal(top_roots) & (top_roots >= 0)].real # keep only real and positive roots
        time_to_top = min(top_roots) if len(top_roots) > 0 else float('inf')
        
        #Bottom Wall 
        bottom_equation = [-0.5*self.g, self.vy, self.y - self.r,]  # coefficients of the quadratic equation
        bottom_roots = np.roots(bottom_equation)
        bottom_roots = bottom_roots[np.isreal(bottom_roots) & (bottom_roots >= 0)].real # keep only real and positive roots
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
        
    def make_grid(self, N: int, particle_radius: float = 1.0):
        """Create a grid of N particles evenly spaced in the box."""
        n_cols = int(math.ceil(math.sqrt(N)))
        n_rows = int(math.ceil(N / n_cols))

        # Use full box, with margin for particle radius
        margin = 2 * particle_radius
        x_array = np.linspace(margin, self.width - margin, n_cols)
        y_array = np.linspace(margin, self.height - margin, n_rows)

        X, Y = np.meshgrid(x_array, y_array)

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
