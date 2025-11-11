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
    particles, box, tracked_heights = simulate(
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
    particles, box, tracked_heights = simulate(
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
    mean_diffs = []
    std_diffs = []
    all_diffs = []  # Store ALL individual differences for hypothesis testing

    def run_trial(args):
        """Helper function for parallel execution."""
        temp = args
        particles, box, tracked_heights = simulate(
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
        return hot_mean, cold_mean

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

        # Store means for correlation
        mean_heights_hot.append(np.mean(hot_heights))
        mean_heights_cold.append(np.mean(cold_heights))

        # Compute differences WITHIN this temperature
        diffs_this_temp = np.array(hot_heights) - np.array(cold_heights)
        mean_diffs.append(np.mean(diffs_this_temp))
        std_diffs.append(np.std(diffs_this_temp) / np.sqrt(ntrials))  # Standard error of mean

        # Store all individual differences for hypothesis testing
        all_diffs.extend(diffs_this_temp)

        print(f"  Mean height hot: {mean_heights_hot[-1]:.2f}")
        print(f"  Mean height cold: {mean_heights_cold[-1]:.2f}")
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


def run_all_experiments():
    """
    Run all experiments in sequence.
    """
    print("\n" + "=" * 60)
    print("MICROHEAT: EXPERIMENTAL SUITE")
    print("=" * 60)
    print()
    print("Testing the geometric collision bias theory")
    print()

    # Run Experiment 1: Equipartition
    experiment_equipartition(N=50, width=3000.0, height=3000.0,
                           temperature=10.0, max_time=50.0, fps=10)

    # Run Experiment 2: One Hot Particle
    experiment_hot_particle(N=50, width=3000.0, height=3000.0,
                          hot_index=5, hot_temperature=500.0,
                          cold_temperature=10.0, max_time=100.0, fps=30)

    # Run Experiment 3: Temperature-Height Correlation
    experiment_temp_height_correlation(
        temp_list=[50, 100, 200, 300, 400, 500],
        hot_index=50,
        cold_temperature=50,
        ntrials=10,
        max_time=100,
        N=100,
        width=1000,
        height=1000
    )

    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_experiments()
