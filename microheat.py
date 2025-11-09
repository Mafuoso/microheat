import numpy as np
import math
import heapq
from tqdm import tqdm
from p_tqdm import p_map


np.random.seed(42)  # For reproducibility
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
        self.heights = [] #to calculate mean position 


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

def get_mean_position(particle:Particle):
    """Calculate the mean height of a particle over its recorded heights."""
    if len(particle.heights) == 0:
        return particle.y
    return sum(particle.heights) / len(particle.heights)


def simulate(hot_index,temp):
    #Experiment Details
    max_time = 1000
    N = 100
    width = 1000
    height = 1000
    particles, box = initialize(N, width, height)
    init_hot_particle(particles, hot_index, hot_temperature=temp, cold_temperature=50)
    events = initialize_events(particles, box)
    current_time = 0
    next_sample = 10 #sample every 10 time units

    pbar = tqdm(total=max_time)
    while current_time < max_time:
        pbar.update(current_time - pbar.n)
        (event_time, i, j, count_i, count_j) = heapq.heappop(events) #Get the next event, if this is a wall collision j = wall name str else j = particle index
        #validitiy check on the event
        if particles[i].collision_count != count_i:
            continue
        if isinstance(j,int) and particles[j].collision_count != count_j:
            continue
        
        while next_sample <= event_time and next_sample <= max_time:
            dt = next_sample - current_time
            if dt > 0:
                advance_particles(particles, dt, box.g)
                current_time = next_sample
            particles[hot_index].heights.append(particles[hot_index].y)
            particles[(hot_index+1)%N].heights.append(particles[(hot_index+1)%N].y) # record height of a cold particle for comparison

            next_sample += 10 #sample interval is 10

        #Advance all particles to event time
        advance_particles(particles, event_time - current_time)
        current_time = event_time
        #Process all collisions
        if isinstance(j, int):  # Particle-Particle collision
            particles[i].collide_with_particle(particles[j])
            predict_new_collisions(particles, i, box, events, current_time)
            predict_new_collisions(particles, j, box, events, current_time)
        else:  # Particle-Wall collision (j is a string)
            particles[i].collide_with_wall(j)
            predict_new_collisions(particles, i, box, events, current_time)

    pbar.close()
    return get_mean_position(particles[hot_index]), get_mean_position(particles[(hot_index+1)%N]) # return mean height of hot particle and a cold particle for comparison


def temp_height_correlate():
    """ Run multiple trials and calculate correlation between temperature and mean height. Fixed hot index"""
    temp_list = [50,100, 200, 300, 400, 500
    ]
    hot_index = 2
 
    mean_heights = p_map(simulate, [hot_index]*len(temp_list), temp_list)
    mean_heights_hot = [h[0] for h in mean_heights]
    mean_heights_cold = [h[1] for h in mean_heights]

    correlation_hot = np.corrcoef(temp_list, mean_heights_hot)[0, 1]
    correlation_cold = np.corrcoef(temp_list, mean_heights_cold)[0, 1]
    return correlation_hot,correlation_cold, temp_list, mean_heights_hot, mean_heights_cold

if __name__ == "__main__":
    correlation_hot, correlation_cold, temp_list, mean_heights_hot, mean_heights_cold = temp_height_correlate()
    print("Correlation between temperature and mean height of hot particle:", correlation_hot)
    print("Correlation between temperature and mean height of cold particle:", correlation_cold)
    for t, h_hot, h_cold in zip(temp_list, mean_heights_hot, mean_heights_cold):
        print(f"Temperature: {t}, Mean Height Hot Particle: {h_hot}, Mean Height Cold Particle: {h_cold}")


