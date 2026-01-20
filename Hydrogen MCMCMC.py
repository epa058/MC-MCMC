import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, genlaguerre
from mpl_toolkits.mplot3d import Axes3D

# Bohr radius (in atomic units)
a0 = 1.0

# Calculate the unnormalized probability density |psi|^2 at (x,y,z)
def get_wavefunction_prob(x, y, z, n, l, m):    
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if r == 0:
        return 0

    # Physics convention because mathematicians are better at abstraction and hence won't get as easily confused
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    # Radial Component
    rho = 2 * r / (n * a0)
    radial_part = np.exp(-rho / 2) * (rho ** l) * genlaguerre(n - l - 1, 2 * l + 1)(rho)

    # Angular Component
    angular_part = sph_harm(m, l, phi, theta)

    psi = radial_part * angular_part
    
    return float(np.abs(psi) ** 2)


def metropolis(n, l, m, num_samples = 50000, step_size = 0.5):
    samples = []
    current_prob = 0

    exp_r = a0 / 2 * (3 * n**2 - l * (l + 1))
    exp_r2 = a0**2 * n**2 / 2 * (5 * n**2 - 3 * l * (l + 1) + 1)
    std_r = np.sqrt(exp_r2 - exp_r**2)

    # Re-sample until we avoid starting at a node
    while current_prob == 0:
        direction = np.random.normal(0, 1, 3)
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue

        # Initialise near the expected orbital radius
        unit_vector = direction / norm
        r_init = np.abs(np.random.normal(exp_r, std_r))
        current_pos = unit_vector * r_init
        current_prob = get_wavefunction_prob(current_pos[0], current_pos[1], current_pos[2], n, l, m)

    for _ in range(num_samples):
        proposal = current_pos + np.random.normal(0, step_size, 3)
        proposal_prob = get_wavefunction_prob(proposal[0], proposal[1], proposal[2], n, l, m)
        if current_prob == 0.0:
            ratio = 1.0
        else:
            ratio = proposal_prob / current_prob
            
        if np.random.rand() < ratio:
            current_pos = proposal
            current_prob = proposal_prob
            
        # Append even if rejected so chain length remains correct
        samples.append(current_pos.copy())
    return np.array(samples)

def parallel_tempering(n, l, m, num_samples = 50000, burn_in = 5000, step_size = 0.5, chains = 8, swap_every = 10, max_temp = 6):
    # ''chains'' independent Markov chains at a range of temperatures between 1 and ''max_temp''
    # Each chain performs a local Metropolis update with a Gaussian proposal of width ''step_size''.
    # After ''swap_every'' steps, adjacent chains propose to swap their configurations.
    # The swaps are accepted according to the appropriate detailed-balance criterion.
    # Samples from the lowest-temperature chain are returned after discarding ''burn_in'' iterations.

    exp_r = a0 / 2 * (3 * n**2 - l * (l + 1))
    exp_r2 = a0**2 * n**2 / 2 * (5 * n**2 - 3 * l * (l + 1) + 1)
    std_r = np.sqrt(exp_r2 - exp_r**2)

    assert chains >= 2, "There must be at least two temperature levels for tempering"
    # Geometric ladder of temperatures: 1 = coldest, max_temp = hottest
    temps = np.geomspace(1.0, max_temp, num=chains)

    positions = np.zeros((chains, 3))
    log_probs = np.zeros(chains, dtype=float)

    for i in range(chains):
        current_prob = 0.0
        
        while current_prob == 0.0:
            # Random Direction
            direction = np.random.normal(0, 1, 3)
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            unit_vector = direction / norm
            
            # Initialise near the expected orbital radius
            r_init = np.abs(np.random.normal(exp_r, std_r))
            current_pos = unit_vector * r_init
            current_prob = get_wavefunction_prob(current_pos[0], current_pos[1], current_pos[2], n, l, m)
            
            positions[i] = current_pos
            log_probs[i] = np.log(current_prob)

    samples = []
    total_iters = burn_in + num_samples
    
    # Each iteration consists of one sweep of local updates across chains
    for t in range(total_iters):
        for i in range(chains):
            current = positions[i]
            current_logprob = log_probs[i]
            
            prop = current + np.random.normal(0.0, step_size, 3)
            prop_prob = get_wavefunction_prob(prop[0], prop[1], prop[2], n, l, m)
            
            if prop_prob > 0.0:
                prop_logprob = np.log(prop_prob)
                
            # Acceptance ratio for tempered target
                if np.log(np.random.rand()) < (prop_logprob - current_logprob) / temps[i]:
                    positions[i] = prop
                    log_probs[i] = prop_logprob
                    
        # Chain Exchange
        if (t + 1) % swap_every == 0:
            # Iterate over neighbouring pairs (0-1, 1-2, ...) and propose swaps
            for _ in range(chains):
                i = np.random.randint(0, chains - 1)
                j = i + 1
                Ti = temps[i]
                Tj = temps[j]
                
                # Compute Metropolis ratio for swapping states i and j
                # Using detailed-balance: exp((1/Ti - 1/Tj) * (logp_j - logp_i))
                if np.log(np.random.rand()) < (1.0 / Ti - 1.0 / Tj) * (log_probs[i + 1] - log_probs[i]):
                    # Swap positions and log probabilities
                    pos_temp = positions[i].copy()
                    positions[i] = positions[j]
                    positions[j] = pos_temp

                    log_prob_temp = log_probs[i]
                    log_probs[i] = log_probs[j]
                    log_probs[j] = log_prob_temp
                    
        # Record sample from coldest chain after burn-in
        if t >= burn_in:
            samples.append(positions[0].copy())
            
    return np.array(samples)


def plot_cloud(samples, n, l, m):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]
    
    # Thin out points to lighten the plot
    thinning = max(1, len(samples) // 50000)
    
    ax.scatter(x[::thinning], y[::thinning], z[::thinning], s=0.5, c='white', alpha=0.1, marker='.')
    ax.set_title(f"Hydrogen Electron Cloud\nState: n={n}, l={l}, m={m}")
    ax.set_xlabel("x ($a_0$)")
    ax.set_ylabel("y ($a_0$)")
    ax.set_zlabel("z ($a_0$)")
    
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.grid(False)

    limit = a0 * (3 * n ** 2 - l * (l + 1))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    plt.show()

if __name__ == "__main__":
    N, L, M = 3, 2, 0
    
    samples = parallel_tempering(N, L, M, num_samples=100000, step_size=0.5, chains=8, swap_every=10, burn_in=5000, max_temp=4)
    plot_cloud(samples, N, L, M)
