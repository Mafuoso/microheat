"""
Experimental configurations for the microheat simulation.

This module contains independent experiment functions that use the simulate()
function from microheat.py along with visualization tools from animate.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from microheat import simulate, initialize, init_hot_particle, init_velocities_equiparition
from animate import animate_simulation

#np.random.seed(42)  # For reproducibility, might need to remove this for statistical experiments

# Try to import p_map for parallel processing, fall back to sequential if not available
try:
    from p_tqdm import p_map
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    print("Warning: p_tqdm not available, using sequential processing for multi-trial experiments")


def plot_height_time_series(times: list, heights_dict: dict,
                            hot_particle_index: int = None,
                            save_file: str = None,
                            title: str = "Particle Height vs Time"):
    """
    Plot height time series for tracked particles.

    Args:
        times: List of time points
        heights_dict: Dict mapping particle_index -> list of heights
        hot_particle_index: If specified, highlights this particle in red
        save_file: If provided, save plot to this file
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each particle's height trajectory
    for particle_idx, height_values in heights_dict.items():
        if particle_idx == hot_particle_index:
            # Plot hot particle in red with thicker line
            ax.plot(times, height_values, color='red', linewidth=2.0,
                   label=f'Hot Particle {particle_idx}', alpha=0.9)
        else:
            # Plot regular particles in blue with thin lines
            ax.plot(times, height_values, color='blue', linewidth=0.5,
                   alpha=0.3, label=f'Particle {particle_idx}' if len(heights_dict) <= 5 else None)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Height (y-position)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    if hot_particle_index is not None or len(heights_dict) <= 5:
        ax.legend()

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"Height time series plot saved to {save_file}")

    plt.close()


def plot_temp_height_correlation(temp_list, mean_heights_hot, mean_heights_cold,
                                 mean_diffs, std_diffs, save_file: str = None):
    """
    Plot the correlation between temperature and height difference.

    Args:
        temp_list: List of temperatures
        mean_heights_hot: Mean heights of hot particle at each temperature
        mean_heights_cold: Mean heights of cold particles at each temperature
        mean_diffs: Mean height differences (hot - cold)
        std_diffs: Standard errors of height differences
        save_file: If provided, save plot to this file
    """
    plt.figure(figsize=(10, 6))
    plt.errorbar(temp_list, mean_diffs, yerr=std_diffs, fmt='o-',
                 label='Mean Height Difference (Hot - Cold)',
                 capsize=5, capthick=2, markersize=8, linewidth=2)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5, label='No difference')
    plt.xlabel('Temperature of Hot Particle', fontsize=12)
    plt.ylabel('Mean Height Difference (Hot - Cold)', fontsize=12)
    plt.title('Mean Height Difference vs Temperature of Hot Particle', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"Temperature-height correlation plot saved to {save_file}")
    else:
        plt.show()

    plt.close()


def plot_distribution_analysis(all_diffs: np.ndarray, shapiro_stat: float, shapiro_p: float,
                               ks_stat: float, ks_p: float, save_file: str = None):
    """
    Plot distribution analysis including histogram, Q-Q plot, and normality test results.

    Args:
        all_diffs: Array of all height differences (hot - cold)
        shapiro_stat: Shapiro-Wilk test statistic
        shapiro_p: Shapiro-Wilk p-value
        ks_stat: Kolmogorov-Smirnov test statistic
        ks_p: Kolmogorov-Smirnov p-value
        save_file: If provided, save plot to this file
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram with normal distribution overlay
    ax1 = axes[0]
    ax1.hist(all_diffs, bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')

    # Fit and plot normal distribution
    mu, sigma = all_diffs.mean(), all_diffs.std()
    x = np.linspace(all_diffs.min(), all_diffs.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')

    ax1.set_xlabel('Height Difference (Hot - Cold)', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_title('Distribution of Height Differences', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Add statistics text box
    textstr = f'μ = {mu:.2f}\nσ = {sigma:.2f}\nn = {len(all_diffs)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Q-Q plot
    ax2 = axes[1]
    stats.probplot(all_diffs, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=12)
    ax2.grid(alpha=0.3)

    # Add normality test results
    test_text = f'Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4f}\n'
    test_text += f'K-S Test: D={ks_stat:.4f}, p={ks_p:.4f}\n'
    if shapiro_p > 0.05 and ks_p > 0.05:
        test_text += 'Result: Normality NOT rejected (α=0.05)'
    else:
        test_text += 'Result: Normality REJECTED (α=0.05)'

    props2 = dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
    ax2.text(0.05, 0.95, test_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=props2)

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"Distribution analysis plot saved to {save_file}")

    plt.close()


def plot_height_with_collisions(times: list, heights: list, collision_events: list,
                                hot_particle_index: int, particle_indices_map: dict,
                                save_file: str = None, title: str = "Height vs Time with Collisions"):
    """
    Plot height vs time with collision markers showing when collisions occur.

    Args:
        times: List of time points for height measurements
        heights: List of heights corresponding to times
        collision_events: List of collision event dictionaries from simulate()
        hot_particle_index: Index of the hot particle
        particle_indices_map: Dict mapping particle indices to descriptions (e.g., {5: 'hot', 6: 'cold'})
        save_file: If provided, save plot to this file
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot height trajectory
    ax.plot(times, heights, 'b-', linewidth=1.5, label='Hot particle height', alpha=0.7)

    # Classify and plot collisions
    collision_with_cold_below = []
    collision_with_cold_above = []
    collision_with_wall = []

    for event in collision_events:
        if event['collision_type'] == 'wall':
            collision_with_wall.append((event['time'], event['height_i']))
        elif event['collision_type'] == 'particle':
            # Determine if hot particle collided with someone below or above
            if event['particle_i'] == hot_particle_index:
                # Hot particle is particle_i
                if event['relative_position'] == 'above':
                    # Hot particle was above the other, so other is below
                    collision_with_cold_below.append((event['time'], event['height_i']))
                elif event['relative_position'] == 'below':
                    # Hot particle was below the other, so other is above
                    collision_with_cold_above.append((event['time'], event['height_i']))
            elif event['particle_j'] == hot_particle_index:
                # Hot particle is particle_j
                if event['relative_position'] == 'above':
                    # particle_i was above particle_j (hot), so hot was below
                    collision_with_cold_above.append((event['time'], event['height_j']))
                elif event['relative_position'] == 'below':
                    # particle_i was below particle_j (hot), so hot was above
                    collision_with_cold_below.append((event['time'], event['height_j']))

    # Plot collision markers
    if collision_with_cold_below:
        times_below, heights_below = zip(*collision_with_cold_below)
        ax.scatter(times_below, heights_below, color='green', s=100, marker='^',
                  label='Collision with cold below (upward boost)', zorder=5, edgecolor='black', linewidth=0.5)

    if collision_with_cold_above:
        times_above, heights_above = zip(*collision_with_cold_above)
        ax.scatter(times_above, heights_above, color='orange', s=100, marker='v',
                  label='Collision with cold above', zorder=5, edgecolor='black', linewidth=0.5)

    if collision_with_wall:
        times_wall, heights_wall = zip(*collision_with_wall)
        ax.scatter(times_wall, heights_wall, color='red', s=60, marker='x',
                  label='Wall collision', zorder=5, linewidth=2)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Height (y-position)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text summary
    total_collisions_below = len(collision_with_cold_below)
    total_collisions_above = len(collision_with_cold_above)
    summary_text = f'Collisions with cold below: {total_collisions_below}\n'
    summary_text += f'Collisions with cold above: {total_collisions_above}\n'
    if total_collisions_below + total_collisions_above > 0:
        ratio = total_collisions_below / (total_collisions_below + total_collisions_above)
        summary_text += f'Ratio (below / total): {ratio:.2f}'

    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"Height with collisions plot saved to {save_file}")

    plt.close()


def experiment_equipartition(N: int = 50, width: float = 3000.0, height: float = 3000.0,
                             temperature: float = 10.0, max_time: float = 50.0,
                             fps: int = 10, k_B: float = 1.0,
                             with_animation: bool = True, with_height_plot: bool = True):
    """
    Experiment 1: Equipartition

    All particles initialized at the same temperature. Tests baseline behavior
    where all particles have similar kinetic energies.

    Args:
        N: Number of particles
        width: Box width
        height: Box height
        temperature: Temperature for all particles
        max_time: Simulation duration
        fps: Frames per second for animation
        k_B: Boltzmann constant
        with_animation: Whether to generate animation
        with_height_plot: Whether to generate height time series plot
    """
    print("=" * 60)
    print("EXPERIMENT 1: EQUIPARTITION")
    print("=" * 60)
    print(f"All particles at temperature T={temperature}")
    print(f"Configuration: {N} particles in {width}x{height} box")
    print()

    # Select particles to track (sample up to 5 particles evenly distributed)
    track_indices = [i * N // 5 for i in range(5) if i * N // 5 < N]
    if len(track_indices) == 0:
        track_indices = [0]

    # Run simulation with tracking
    print("Running simulation with height tracking...")
    sample_interval = 1.0
    particles, box, tracked_heights, _ = simulate(
        hot_index=None,  # No hot particle for equipartition
        cold_temperature=temperature,
        max_time=max_time,
        N=N,
        width=width,
        height=height,
        sample_interval=sample_interval,
        track_indices=track_indices,
        show_progress=True,
        k_B=k_B
    )

    # Generate time points for plotting (match the sampling points)
    times = [sample_interval * i for i in range(len(list(tracked_heights.values())[0]))]

    # Plot height time series
    if with_height_plot:
        plot_height_time_series(times, tracked_heights,
                               save_file="experiment1_height_timeseries.png",
                               title=f"Equipartition: Particle Heights (T={temperature})")

    # Create animation
    if with_animation:
        print("Generating animation...")
        particles_anim, box_anim = initialize(N=N, width=width, height=height)
        init_velocities_equiparition(particles_anim, temperature=temperature, k_B=k_B)

        animate_simulation(particles_anim, box_anim,
                          max_time=max_time, fps=fps,
                          save_file="experiment1_equipartition_animation.gif",
                          title=f"Equipartition: T={temperature}")

    print()
    print("Experiment 1 complete!")
    print()

    return particles, box, tracked_heights


def experiment_hot_particle(N: int = 50, width: float = 3000.0, height: float = 3000.0,
                           hot_index: int = 5, hot_temperature: float = 500.0,
                           cold_temperature: float = 10.0, max_time: float = 100.0,
                           fps: int = 30, k_B: float = 1.0,
                           with_animation: bool = True, with_height_plot: bool = True):
    """
    Experiment 2: One Hot Particle

    Tests the geometric collision bias theory by initializing one particle with
    much higher temperature than the others. Tracks whether the hot particle
    exhibits elevated height over time.

    Args:
        N: Number of particles
        width: Box width
        height: Box height
        hot_index: Index of the hot particle
        hot_temperature: Temperature of the hot particle
        cold_temperature: Temperature of the cold particles
        max_time: Simulation duration
        fps: Frames per second for animation
        k_B: Boltzmann constant
        with_animation: Whether to generate animation
        with_height_plot: Whether to generate height time series plot
    """
    print("=" * 60)
    print("EXPERIMENT 2: ONE HOT PARTICLE")
    print("=" * 60)
    print(f"Hot particle at index {hot_index}: T={hot_temperature}")
    print(f"All other particles: T={cold_temperature}")
    print(f"Configuration: {N} particles in {width}x{height} box")
    print()

    # Select particles to track (hot particle + a few cold ones)
    other_indices = [i * N // 5 for i in range(5) if i * N // 5 < N and i * N // 5 != hot_index]
    track_indices = [hot_index] + other_indices[:4]

    # Run simulation with tracking
    print("Running simulation with height tracking...")
    sample_interval = 1.0
    particles, box, tracked_heights, _ = simulate(
        hot_index=hot_index,
        hot_temperature=hot_temperature,
        cold_temperature=cold_temperature,
        max_time=max_time,
        N=N,
        width=width,
        height=height,
        sample_interval=sample_interval,
        track_indices=track_indices,
        show_progress=True,
        k_B=k_B
    )

    # Generate time points for plotting (match the sampling points)
    times = [sample_interval * i for i in range(len(list(tracked_heights.values())[0]))]

    # Plot height time series with hot particle highlighted
    if with_height_plot:
        plot_height_time_series(times, tracked_heights,
                               hot_particle_index=hot_index,
                               save_file="experiment2_height_timeseries.png",
                               title=f"Hot Particle: Heights (T_hot={hot_temperature}, T_cold={cold_temperature})")

    # Create animation with hot particle highlighted
    if with_animation:
        print("Generating animation...")
        particles_anim, box_anim = initialize(N=N, width=width, height=height)
        init_hot_particle(particles_anim, hot_index=hot_index,
                         hot_temperature=hot_temperature,
                         cold_temperature=cold_temperature, k_B=k_B)

        animate_simulation(particles_anim, box_anim,
                          max_time=max_time, fps=fps,
                          save_file="experiment2_hot_particle_animation.gif",
                          title=f"Hot Particle: T_hot={hot_temperature}, T_cold={cold_temperature}",
                          hot_particle_index=hot_index)

    print()
    print("Experiment 2 complete!")
    print()

    return particles, box, tracked_heights


def experiment_temp_height_correlation(temp_list: list[float] = None,
                                       hot_index: int = 50,
                                       cold_temperature: float = 50,
                                       ntrials: int = 10,
                                       max_time: float = 100,
                                       N: int = 100,
                                       width: float = 1000,
                                       height: float = 1000):
    """
    Experiment 3: Temperature-Height Correlation

    Run multiple trials at different temperatures to analyze correlation between
    hot particle temperature and mean height difference.

    Args:
        temp_list: List of temperatures to test
        hot_index: Index of hot particle
        cold_temperature: Temperature of cold particles
        ntrials: Number of trials per temperature
        max_time: Simulation duration per trial
        N: Number of particles
        width: Box width
        height: Box height

    Returns:
        dict with correlation results
    """
    if temp_list is None:
        temp_list = [50, 100, 200, 300, 400, 500]

    print("=" * 60)
    print("EXPERIMENT 3: TEMPERATURE-HEIGHT CORRELATION")
    print("=" * 60)
    print(f"Testing temperatures: {temp_list}")
    print(f"Number of trials per temperature: {ntrials}")
    print(f"Configuration: {N} particles in {width}x{height} box")
    print(f"Hot particle index: {hot_index}, Cold temperature: {cold_temperature}")
    print()

    mean_heights_hot = []
    mean_heights_cold = []
    std_heights_hot = []   
    std_heights_cold = []
    mean_diffs = []
    std_diffs = []
    all_diffs = []  # Store ALL individual differences for hypothesis testing

    def run_trial(args):
        """Helper function for parallel execution."""
        temp = args
        particles, box, tracked_heights, _ = simulate(
            hot_index=hot_index,
            hot_temperature=temp,
            cold_temperature=cold_temperature,
            max_time=max_time,
            N=N,
            width=width,
            height=height,
            sample_interval=10,
            track_indices=[hot_index, (hot_index + 1) % N],
            show_progress=False
        )
        # Calculate mean heights from tracked data
        hot_mean = np.mean(tracked_heights[hot_index])
        cold_mean = np.mean(tracked_heights[(hot_index + 1) % N])
        hot_std = np.std(tracked_heights[hot_index])
        cold_std = np.std(tracked_heights[(hot_index + 1) % N])
         
        
        return hot_mean, cold_mean, hot_std, cold_std 

    for temp in temp_list:
        print(f"\nTesting temperature T={temp}...")
        # Run ntrials simulations for this temperature
        if PARALLEL_AVAILABLE:
            results = p_map(run_trial, [temp] * ntrials)
        else:
            # Fall back to sequential processing with progress bar
            results = []
            for _ in tqdm(range(ntrials), desc=f"T={temp}"):
                results.append(run_trial(temp))

        # Extract hot and cold heights for THIS temperature
        hot_heights = [r[0] for r in results]
        cold_heights = [r[1] for r in results]
        hot_stds = [r[2] for r in results]
        cold_stds = [r[3] for r in results]

        # Store means for correlation
        mean_heights_hot.append(np.mean(hot_heights))
        mean_heights_cold.append(np.mean(cold_heights))
        std_heights_cold.append(np.std(cold_heights) / np.sqrt(ntrials))
        std_heights_hot.append(np.std(hot_heights) / np.sqrt(ntrials))  # Standard error of mean for individuals and not differences
        # Compute differences WITHIN this temperature
        diffs_this_temp = np.array(hot_heights) - np.array(cold_heights)
        mean_diffs.append(np.mean(diffs_this_temp))
        std_diffs.append(np.std(diffs_this_temp) / np.sqrt(ntrials))  # Standard error of mean, why is this 0? 

        # Store all individual differences for hypothesis testing
        all_diffs.extend(diffs_this_temp)

        print(f"  Mean height hot: {mean_heights_hot[-1]:.2f} +/- {std_heights_hot[-1]:.2f}")
        print(f"  Mean height cold: {mean_heights_cold[-1]:.2f} +/- {std_heights_cold[-1]:.2f}")
        print(f"  Mean difference: {mean_diffs[-1]:.2f} ± {std_diffs[-1]:.2f}")

    # Convert to numpy array for statistical tests
    all_diffs = np.array(all_diffs)

    correlation_hot = np.corrcoef(temp_list, mean_heights_hot)[0, 1]
    correlation_cold = np.corrcoef(temp_list, mean_heights_cold)[0, 1]

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Correlation (temperature vs hot particle height): {correlation_hot:.4f}")
    print(f"Correlation (temperature vs cold particle height): {correlation_cold:.4f}")
    print()

    # ========================================================================
    # HYPOTHESIS TESTING
    # ========================================================================
    print("=" * 60)
    print("HYPOTHESIS TESTING")
    print("=" * 60)
    print()

    # 1. Goodness of Fit Tests (test for normality)
    print("1. GOODNESS OF FIT TESTS (Normality)")
    print("-" * 60)

    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(all_diffs)
    print(f"Shapiro-Wilk Test:")
    print(f"  Test statistic (W): {shapiro_stat:.6f}")
    print(f"  p-value: {shapiro_p:.6f}")
    if shapiro_p > 0.05:
        print(f"  Conclusion: Fail to reject normality (p > 0.05)")
        print(f"              Data is consistent with normal distribution")
    else:
        print(f"  Conclusion: Reject normality (p ≤ 0.05)")
        print(f"              Data deviates from normal distribution")
    print()

    # Kolmogorov-Smirnov test for normality
    # Fit normal distribution to the data
    mu_fit, sigma_fit = all_diffs.mean(), all_diffs.std()
    ks_stat, ks_p = stats.kstest(all_diffs, lambda x: stats.norm.cdf(x, mu_fit, sigma_fit))
    print(f"Kolmogorov-Smirnov Test (vs Normal distribution):")
    print(f"  Test statistic (D): {ks_stat:.6f}")
    print(f"  p-value: {ks_p:.6f}")
    if ks_p > 0.05:
        print(f"  Conclusion: Fail to reject normality (p > 0.05)")
        print(f"              Data is consistent with normal distribution")
    else:
        print(f"  Conclusion: Reject normality (p ≤ 0.05)")
        print(f"              Data deviates from normal distribution")
    print()

    # 2. Population Mean Hypothesis Test
    print("2. POPULATION MEAN HYPOTHESIS TEST")
    print("-" * 60)
    print("H₀: μ_diff = 0  (no height difference between hot and cold particles)")
    print("H₁: μ_diff > 0  (hot particles have greater mean height)")
    print()

    # One-sample t-test (one-tailed)
    # Testing if mean difference > 0
    t_stat, p_value_two_tail = stats.ttest_1samp(all_diffs, 0)
    p_value_one_tail = p_value_two_tail / 2  # Convert to one-tailed

    # For one-tailed test, we only care about positive differences
    if t_stat > 0:
        p_value_one_tail = p_value_two_tail / 2
    else:
        p_value_one_tail = 1 - (p_value_two_tail / 2)

    print(f"One-Sample t-test (one-tailed, α=0.05):")
    print(f"  Sample mean: {all_diffs.mean():.4f}")
    print(f"  Sample std: {all_diffs.std():.4f}")
    print(f"  Sample size: {len(all_diffs)}")
    print(f"  t-statistic: {t_stat:.6f}")
    print(f"  p-value (one-tailed): {p_value_one_tail:.6f}")
    print()

    if p_value_one_tail < 0.05:
        print(f"  Conclusion: REJECT H₀ (p < 0.05)")
        print(f"              Strong evidence that hot particles have greater mean height")
    else:
        print(f"  Conclusion: FAIL TO REJECT H₀ (p ≥ 0.05)")
        print(f"              Insufficient evidence that hot particles have greater mean height")
    print()

    # Calculate 95% confidence interval for the mean difference
    ci_95 = stats.t.interval(0.95, len(all_diffs)-1,
                             loc=all_diffs.mean(),
                             scale=stats.sem(all_diffs))
    print(f"  95% Confidence Interval for μ_diff: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    print()

    # Effect size (Cohen's d)
    cohens_d = all_diffs.mean() / all_diffs.std()
    print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        effect_desc = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_desc = "small"
    elif abs(cohens_d) < 0.8:
        effect_desc = "medium"
    else:
        effect_desc = "large"
    print(f"  Effect size interpretation: {effect_desc}")
    print()

    # Plot distribution analysis
    plot_distribution_analysis(all_diffs, shapiro_stat, shapiro_p, ks_stat, ks_p,
                               save_file="experiment3_distribution_analysis.png")

    # Plot correlation results
    plot_temp_height_correlation(temp_list, mean_heights_hot, mean_heights_cold,
                                mean_diffs, std_diffs,
                                save_file="experiment3_temp_height_correlation.png")

    print("=" * 60)
    print("Experiment 3 complete!")
    print("=" * 60)
    print()

    return {
        'correlation_hot': correlation_hot,
        'correlation_cold': correlation_cold,
        'temp_list': temp_list,
        'mean_heights_hot': mean_heights_hot,
        'mean_heights_cold': mean_heights_cold,
        'mean_diffs': mean_diffs,
        'std_diffs': std_diffs,
        'all_diffs': all_diffs,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        't_stat': t_stat,
        'p_value_one_tail': p_value_one_tail,
        'ci_95': ci_95,
        'cohens_d': cohens_d
    }


def experiment_collision_tracking(N: int = 50, width: float = 3000.0, height: float = 3000.0,
                                 hot_index: int = 5, hot_temperature: float = 500.0,
                                 cold_temperature: float = 10.0, max_time: float = 100.0,
                                 k_B: float = 1.0):
    """
    Experiment 4: Collision Tracking

    Track collisions involving the hot particle and visualize height trajectory
    with collision markers. This shows that rises in height correspond with
    collisions with cold particles below.

    Args:
        N: Number of particles
        width: Box width
        height: Box height
        hot_index: Index of the hot particle
        hot_temperature: Temperature of the hot particle
        cold_temperature: Temperature of the cold particles
        max_time: Simulation duration
        k_B: Boltzmann constant
    """
    print("=" * 60)
    print("EXPERIMENT 4: COLLISION TRACKING")
    print("=" * 60)
    print(f"Hot particle at index {hot_index}: T={hot_temperature}")
    print(f"All other particles: T={cold_temperature}")
    print(f"Configuration: {N} particles in {width}x{height} box")
    print(f"Tracking collisions to correlate with height changes")
    print()

    # Run simulation with collision tracking
    print("Running simulation with collision tracking...")
    sample_interval = 0.5  # More frequent sampling for detailed trajectory
    particles, box, tracked_heights, collision_events = simulate(
        hot_index=hot_index,
        hot_temperature=hot_temperature,
        cold_temperature=cold_temperature,
        max_time=max_time,
        N=N,
        width=width,
        height=height,
        sample_interval=sample_interval,
        track_indices=[hot_index],
        show_progress=True,
        k_B=k_B,
        track_collisions=True
    )

    # Generate time points for plotting
    times = [sample_interval * i for i in range(len(tracked_heights[hot_index]))]
    heights = tracked_heights[hot_index]

    # Create particle index map
    particle_map = {hot_index: 'hot'}
    for i in range(N):
        if i != hot_index:
            particle_map[i] = 'cold'

    # Plot height with collision markers
    plot_height_with_collisions(
        times, heights, collision_events, hot_index, particle_map,
        save_file="experiment4_collision_tracking.png",
        title=f"Hot Particle Height with Collision Events (T_hot={hot_temperature}, T_cold={cold_temperature})"
    )

    # Analyze collision statistics
    collisions_below = 0
    collisions_above = 0

    for event in collision_events:
        if event['collision_type'] == 'particle':
            if event['particle_i'] == hot_index:
                if event['relative_position'] == 'above':
                    collisions_below += 1
                elif event['relative_position'] == 'below':
                    collisions_above += 1
            elif event['particle_j'] == hot_index:
                if event['relative_position'] == 'above':
                    collisions_above += 1
                elif event['relative_position'] == 'below':
                    collisions_below += 1

    total_particle_collisions = collisions_below + collisions_above

    print("\n" + "=" * 60)
    print("COLLISION ANALYSIS")
    print("=" * 60)
    print(f"Total particle collisions involving hot particle: {total_particle_collisions}")
    print(f"  Collisions with cold particle below: {collisions_below}")
    print(f"  Collisions with cold particle above: {collisions_above}")
    if total_particle_collisions > 0:
        ratio = collisions_below / total_particle_collisions
        print(f"  Ratio (below / total): {ratio:.3f}")
        print()
        if ratio > 0.5:
            print("  Result: Hot particle experiences MORE collisions with cold particles below")
            print("          This supports the geometric collision bias theory!")
        else:
            print("  Result: No significant bias toward collisions below")
    print()

    print("Experiment 4 complete!")
    print()

    return {
        'particles': particles,
        'box': box,
        'tracked_heights': tracked_heights,
        'collision_events': collision_events,
        'collisions_below': collisions_below,
        'collisions_above': collisions_above
    }


def experiment_control_no_gravity(N: int = 50, width: float = 3000.0, height: float = 3000.0,
                                   hot_index: int = 5, hot_temperature: float = 500.0,
                                   cold_temperature: float = 10.0, max_time: float = 100.0,
                                   k_B: float = 1.0):
    """
    Experiment 5: Control - No Gravity (g=0)

    Control experiment with gravity set to zero. Without gravity, there should be
    no height stratification or bias in collision geometry. This tests whether the
    observed effects in normal experiments are truly due to gravity-induced
    ballistic stratification.

    Args:
        N: Number of particles
        width: Box width
        height: Box height
        hot_index: Index of the hot particle
        hot_temperature: Temperature of the hot particle
        cold_temperature: Temperature of the cold particles
        max_time: Simulation duration
        k_B: Boltzmann constant
    """
    print("=" * 60)
    print("EXPERIMENT 5: CONTROL - NO GRAVITY (g=0)")
    print("=" * 60)
    print(f"Hot particle at index {hot_index}: T={hot_temperature}")
    print(f"All other particles: T={cold_temperature}")
    print(f"Configuration: {N} particles in {width}x{height} box")
    print(f"GRAVITY SET TO ZERO - control experiment")
    print()

    # Run simulation with collision tracking and g=0
    print("Running simulation with g=0 and collision tracking...")
    sample_interval = 0.5
    particles, box, tracked_heights, collision_events = simulate(
        hot_index=hot_index,
        hot_temperature=hot_temperature,
        cold_temperature=cold_temperature,
        max_time=max_time,
        N=N,
        width=width,
        height=height,
        sample_interval=sample_interval,
        track_indices=[hot_index],
        show_progress=True,
        k_B=k_B,
        gravity=0.0,  # NO GRAVITY
        track_collisions=True
    )

    # Generate time points for plotting
    times = [sample_interval * i for i in range(len(tracked_heights[hot_index]))]
    heights = tracked_heights[hot_index]

    # Create particle index map
    particle_map = {hot_index: 'hot'}
    for i in range(N):
        if i != hot_index:
            particle_map[i] = 'cold'

    # Plot height with collision markers
    plot_height_with_collisions(
        times, heights, collision_events, hot_index, particle_map,
        save_file="experiment5_control_no_gravity.png",
        title=f"Control (g=0): Hot Particle Height with Collisions (T_hot={hot_temperature}, T_cold={cold_temperature})"
    )

    # Analyze collision statistics
    collisions_below = 0
    collisions_above = 0

    for event in collision_events:
        if event['collision_type'] == 'particle':
            if event['particle_i'] == hot_index:
                if event['relative_position'] == 'above':
                    collisions_below += 1
                elif event['relative_position'] == 'below':
                    collisions_above += 1
            elif event['particle_j'] == hot_index:
                if event['relative_position'] == 'above':
                    collisions_above += 1
                elif event['relative_position'] == 'below':
                    collisions_below += 1

    total_particle_collisions = collisions_below + collisions_above

    print("\n" + "=" * 60)
    print("COLLISION ANALYSIS (CONTROL - g=0)")
    print("=" * 60)
    print(f"Total particle collisions involving hot particle: {total_particle_collisions}")
    print(f"  Collisions with cold particle below: {collisions_below}")
    print(f"  Collisions with cold particle above: {collisions_above}")
    if total_particle_collisions > 0:
        ratio = collisions_below / total_particle_collisions
        print(f"  Ratio (below / total): {ratio:.3f}")
        print()
        if abs(ratio - 0.5) < 0.1:
            print("  Result: No significant bias (ratio ≈ 0.5)")
            print("          This confirms gravity is necessary for the collision bias!")
        else:
            print(f"  Result: Unexpected bias even without gravity (ratio = {ratio:.3f})")
    print()

    print("Experiment 5 complete!")
    print()

    return {
        'particles': particles,
        'box': box,
        'tracked_heights': tracked_heights,
        'collision_events': collision_events,
        'collisions_below': collisions_below,
        'collisions_above': collisions_above
    }


def experiment_definitive_collision_bias_test(
    T_hot: float = 500.0,
    T_cold: float = 50.0,
    max_time: float = 10.0,
    ntrials: int = 20,
    k_B: float = 1.0
):
    """
    Experiment 6: DEFINITIVE COLLISION BIAS TEST

    The critical experiment: Does elevation depend on collision rate at constant temperature?

    Tests three scenarios:
    1. Pure ballistics (N=1, no collisions possible)
    2. Sparse system (N=100, box=1000, Kn~0.56, few collisions)
    3. Dense system (N=500, box=500, Kn~0.1, many collisions)

    If collision bias is real: elevation should INCREASE with density (more collisions)
    If only ballistics: elevation should be CONSTANT (h ~ v²/2g independent of collisions)

    Args:
        T_hot: Temperature of hot particle
        T_cold: Temperature of cold particles
        max_time: Simulation duration
        ntrials: Number of trials per configuration
        k_B: Boltzmann constant
    """
    print("=" * 80)
    print("EXPERIMENT 6: DEFINITIVE COLLISION BIAS TEST")
    print("=" * 80)
    print("\nThis experiment tests whether collision rate affects elevation")
    print("at constant temperature. This distinguishes pure ballistics from")
    print("collision bias mechanism.\n")
    print(f"Temperature: T_hot={T_hot}, T_cold={T_cold}")
    print(f"Trials per configuration: {ntrials}")
    print(f"Simulation time: {max_time} seconds")
    print()

    configurations = [
        {
            'name': 'Pure Ballistics (N=1)',
            'N': 1,
            'width': 1000,
            'height': 1000,
            'description': 'Single particle - NO collisions possible',
            'expected_Kn': 'N/A',
            'expected_collisions': 0,
        },
        {
            'name': 'Sparse (Current Setup)',
            'N': 100,
            'width': 1000,
            'height': 1000,
            'description': 'Your current configuration',
            'expected_Kn': 0.56,
            'expected_collisions': '~0-2 per particle',
        },
        {
            'name': 'Dense',
            'N': 500,
            'width': 500,
            'height': 500,
            'description': 'High collision rate',
            'expected_Kn': 0.1,
            'expected_collisions': '~5-20 per particle',
        },
    ]

    results = []

    for config in configurations:
        print("=" * 80)
        print(f"TESTING: {config['name']}")
        print("=" * 80)
        print(f"Description: {config['description']}")
        print(f"Configuration: N={config['N']}, box={config['width']}×{config['height']}")
        print(f"Expected Kn: {config['expected_Kn']}")
        print(f"Expected collisions: {config['expected_collisions']}")
        print()

        elevations = []
        collision_counts = []
        collisions_below_counts = []
        collisions_above_counts = []

        # Run multiple trials
        if PARALLEL_AVAILABLE and ntrials > 5:
            print(f"Running {ntrials} trials in parallel...")

            def run_trial_wrapper(trial_num):
                # Determine indices
                hot_idx = 0
                cold_idx = 1 if config['N'] > 1 else None
                track_indices = [hot_idx] if cold_idx is None else [hot_idx, cold_idx]

                # Run simulation
                particles, box, tracked_heights, collision_events = simulate(
                    hot_index=hot_idx if cold_idx is not None else None,
                    hot_temperature=T_hot,
                    cold_temperature=T_cold,
                    max_time=max_time,
                    N=config['N'],
                    width=config['width'],
                    height=config['height'],
                    sample_interval=1.0,
                    track_indices=track_indices,
                    show_progress=False,
                    k_B=k_B,
                    track_collisions=True
                )

                # Calculate elevation
                hot_mean = np.mean(tracked_heights[hot_idx])
                cold_mean = np.mean(tracked_heights[cold_idx]) if cold_idx is not None else config['height']/2
                elevation = hot_mean - cold_mean

                # Count collisions
                total_collisions = len([e for e in collision_events if e['collision_type'] == 'particle']) if collision_events else 0

                # Count collision geometry
                below = 0
                above = 0
                if collision_events:
                    for event in collision_events:
                        if event['collision_type'] == 'particle':
                            if event['particle_i'] == hot_idx:
                                if event['relative_position'] == 'above':
                                    below += 1
                                elif event['relative_position'] == 'below':
                                    above += 1
                            elif event['particle_j'] == hot_idx:
                                if event['relative_position'] == 'above':
                                    above += 1
                                elif event['relative_position'] == 'below':
                                    below += 1

                return elevation, total_collisions, below, above

            trial_results = p_map(run_trial_wrapper, range(ntrials))

            for elevation, total_coll, below, above in trial_results:
                elevations.append(elevation)
                collision_counts.append(total_coll)
                collisions_below_counts.append(below)
                collisions_above_counts.append(above)

        else:
            print(f"Running {ntrials} trials sequentially...")
            for trial in tqdm(range(ntrials), desc=f"{config['name']}"):
                # Determine indices
                hot_idx = 0
                cold_idx = 1 if config['N'] > 1 else None
                track_indices = [hot_idx] if cold_idx is None else [hot_idx, cold_idx]

                # Run simulation
                particles, box, tracked_heights, collision_events = simulate(
                    hot_index=hot_idx if cold_idx is not None else None,
                    hot_temperature=T_hot,
                    cold_temperature=T_cold,
                    max_time=max_time,
                    N=config['N'],
                    width=config['width'],
                    height=config['height'],
                    sample_interval=1.0,
                    track_indices=track_indices,
                    show_progress=False,
                    k_B=k_B,
                    track_collisions=True
                )

                # Calculate elevation
                hot_mean = np.mean(tracked_heights[hot_idx])
                cold_mean = np.mean(tracked_heights[cold_idx]) if cold_idx is not None else config['height']/2
                elevations.append(hot_mean - cold_mean)

                # Count collisions
                total_collisions = len([e for e in collision_events if e['collision_type'] == 'particle']) if collision_events else 0
                collision_counts.append(total_collisions)

                # Count collision geometry
                below = 0
                above = 0
                if collision_events:
                    for event in collision_events:
                        if event['collision_type'] == 'particle':
                            if event['particle_i'] == hot_idx:
                                if event['relative_position'] == 'above':
                                    below += 1
                                elif event['relative_position'] == 'below':
                                    above += 1
                            elif event['particle_j'] == hot_idx:
                                if event['relative_position'] == 'above':
                                    above += 1
                                elif event['relative_position'] == 'below':
                                    below += 1

                collisions_below_counts.append(below)
                collisions_above_counts.append(above)

        # Store results
        result = {
            'config': config,
            'mean_elevation': np.mean(elevations),
            'std_elevation': np.std(elevations),
            'sem_elevation': np.std(elevations) / np.sqrt(ntrials),
            'mean_collisions': np.mean(collision_counts),
            'mean_collisions_below': np.mean(collisions_below_counts),
            'mean_collisions_above': np.mean(collisions_above_counts),
            'elevations': elevations,
        }
        results.append(result)

        print(f"\nRESULTS for {config['name']}:")
        print(f"  Mean elevation: {result['mean_elevation']:.2f} ± {result['sem_elevation']:.2f}")
        print(f"  Mean collisions (hot particle): {result['mean_collisions']:.2f}")
        print(f"    Collisions below: {result['mean_collisions_below']:.2f}")
        print(f"    Collisions above: {result['mean_collisions_above']:.2f}")
        if result['mean_collisions'] > 0:
            ratio = result['mean_collisions_below'] / (result['mean_collisions_below'] + result['mean_collisions_above'])
            print(f"    Ratio (below/total): {ratio:.3f}")
        print()

    # ========================================================================
    # STATISTICAL COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("DEFINITIVE TEST RESULTS")
    print("=" * 80)
    print()

    # Extract results
    pure_ballistics = results[0]
    sparse = results[1]
    dense = results[2]

    print("ELEVATION SUMMARY:")
    print(f"  Pure Ballistics (N=1):  {pure_ballistics['mean_elevation']:8.2f} ± {pure_ballistics['sem_elevation']:.2f}")
    print(f"  Sparse (Kn~0.5):        {sparse['mean_elevation']:8.2f} ± {sparse['sem_elevation']:.2f}")
    print(f"  Dense (Kn~0.1):         {dense['mean_elevation']:8.2f} ± {dense['sem_elevation']:.2f}")
    print()

    # Statistical tests
    print("STATISTICAL TESTS:")
    print("-" * 80)

    # Test 1: Sparse vs Pure Ballistics
    t_stat_1, p_value_1 = stats.ttest_ind(sparse['elevations'], pure_ballistics['elevations'])
    print(f"\n1. Sparse vs Pure Ballistics (t-test):")
    print(f"   t-statistic: {t_stat_1:.4f}")
    print(f"   p-value: {p_value_1:.6f}")
    if p_value_1 > 0.05:
        print(f"   → NO significant difference (p > 0.05)")
        print(f"   → Sparse system elevation is PURE BALLISTICS!")
    else:
        print(f"   → Significant difference (p < 0.05)")

    # Test 2: Dense vs Sparse
    t_stat_2, p_value_2 = stats.ttest_ind(dense['elevations'], sparse['elevations'])
    print(f"\n2. Dense vs Sparse (t-test):")
    print(f"   t-statistic: {t_stat_2:.4f}")
    print(f"   p-value: {p_value_2:.6f}")
    if p_value_2 < 0.05 and dense['mean_elevation'] > sparse['mean_elevation']:
        print(f"   → Dense SIGNIFICANTLY HIGHER (p < 0.05)")
        print(f"   → COLLISION BIAS CONFIRMED!")
    elif p_value_2 > 0.05:
        print(f"   → NO significant difference (p > 0.05)")
        print(f"   → No collision bias effect")
    else:
        print(f"   → Dense LOWER than sparse")
        print(f"   → Thermalization dominates")

    # Test 3: Dense vs Pure Ballistics
    t_stat_3, p_value_3 = stats.ttest_ind(dense['elevations'], pure_ballistics['elevations'])
    print(f"\n3. Dense vs Pure Ballistics (t-test):")
    print(f"   t-statistic: {t_stat_3:.4f}")
    print(f"   p-value: {p_value_3:.6f}")

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    # Interpret results
    sparse_vs_pure = abs(sparse['mean_elevation'] - pure_ballistics['mean_elevation']) / pure_ballistics['sem_elevation']
    dense_vs_sparse = (dense['mean_elevation'] - sparse['mean_elevation']) / sparse['sem_elevation']

    if sparse_vs_pure < 2:  # Less than 2 standard errors difference
        print("✓ FINDING 1: Sparse ≈ Pure Ballistics")
        print("  → Your current system (N=100, box=1000) has TOO FEW collisions")
        print("  → Elevation is from ballistics alone (h ~ v²/2g)")
        print("  → NOT testing collision bias mechanism!")
        print()

    if dense_vs_sparse > 2 and dense['mean_elevation'] > sparse['mean_elevation']:
        print("✓ FINDING 2: Dense > Sparse (SIGNIFICANT)")
        print("  → More collisions → MORE elevation")
        print("  → COLLISION BIAS MECHANISM CONFIRMED!")
        print("  → Geometric bias is REAL and measurable!")
        print()
    elif abs(dense_vs_sparse) < 2:
        print("✗ FINDING 2: Dense ≈ Sparse")
        print("  → Collision rate doesn't affect elevation")
        print("  → NO collision bias - only pure ballistics")
        print()
    else:
        print("✗ FINDING 2: Dense < Sparse")
        print("  → More collisions → LESS elevation")
        print("  → Thermalization overwhelms any collision bias")
        print()

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Elevation vs Configuration
    ax1 = axes[0]
    x_pos = [0, 1, 2]
    elevations_mean = [r['mean_elevation'] for r in results]
    elevations_sem = [r['sem_elevation'] for r in results]
    labels = ['Pure\nBallistics\n(N=1)', 'Sparse\n(Kn~0.5)\nN=100', 'Dense\n(Kn~0.1)\nN=500']
    colors = ['gray', 'orange', 'red']

    bars = ax1.bar(x_pos, elevations_mean, yerr=elevations_sem,
                   color=colors, alpha=0.7, capsize=10, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('Mean Elevation (Hot - Cold)', fontsize=12)
    ax1.set_title('Elevation vs Collision Rate', fontsize=14, fontweight='bold')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='y', alpha=0.3)

    # Add significance stars
    if p_value_2 < 0.05:
        y_max = max(elevations_mean) + max(elevations_sem) + 10
        ax1.plot([1, 2], [y_max, y_max], 'k-', linewidth=1.5)
        significance = '***' if p_value_2 < 0.001 else ('**' if p_value_2 < 0.01 else '*')
        ax1.text(1.5, y_max + 5, significance, ha='center', fontsize=16, fontweight='bold')

    # Plot 2: Collision Counts
    ax2 = axes[1]
    collision_means = [r['mean_collisions'] for r in results]
    ax2.bar(x_pos, collision_means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Mean Collisions (Hot Particle)', fontsize=12)
    ax2.set_title('Collision Rate by Configuration', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Collision Geometry Ratio
    ax3 = axes[2]
    ratios = []
    for r in results[1:]:  # Skip N=1 (no collisions)
        total = r['mean_collisions_below'] + r['mean_collisions_above']
        ratio = r['mean_collisions_below'] / total if total > 0 else 0.5
        ratios.append(ratio)

    x_pos_ratios = [1, 2]
    ax3.bar(x_pos_ratios, ratios, color=colors[1:], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.axhline(0.5, color='red', linestyle='--', linewidth=2, label='No bias (0.5)')
    ax3.set_xticks(x_pos_ratios)
    ax3.set_xticklabels(labels[1:], fontsize=10)
    ax3.set_ylabel('Collision Ratio (Below / Total)', fontsize=12)
    ax3.set_title('Geometric Collision Bias', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment6_definitive_test.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: experiment6_definitive_test.png")
    plt.close()

    print()
    print("=" * 80)
    print("Experiment 6 complete!")
    print("=" * 80)
    print()

    return {
        'results': results,
        'pure_ballistics': pure_ballistics,
        'sparse': sparse,
        'dense': dense,
        'p_sparse_vs_pure': p_value_1,
        'p_dense_vs_sparse': p_value_2,
        'p_dense_vs_pure': p_value_3,
    }


def run_all_experiments():
    """
    Run all experiments in sequence.
    """
   
    # # Run Experiment 1: Equipartition
    # experiment_equipartition(N=50, width=3000.0, height=3000.0,
    #                        temperature=10.0, max_time=50.0, fps=10)

    # # Run Experiment 2: One Hot Particle
    # experiment_hot_particle(N=50, width=3000.0, height=3000.0,
    #                       hot_index=5, hot_temperature=500.0,
    #                       cold_temperature=10.0, max_time=100.0, fps=30)

    # Run Experiment 3: Temperature-Height Correlation
    experiment_temp_height_correlation(
        temp_list= [50,100,150,200,250,300,350,400,450,500],
        hot_index=50,
        cold_temperature=50,
        ntrials=10,
        max_time=10,
        N=100,
        width=1000,
        height=1000
    )

    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_experiments()