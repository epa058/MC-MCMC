
# (Metropolis Coupling) Monte Carlo Markov Chain

I didn't know what MCMCs were, but now I do.

This project visualizes the electron orbitals of a Hydrogen atom by sampling the wavefunction probability density $ |\psi_{nlm}(r, \theta, \phi)|^2 $ using Markov Chain Monte Carlo (MCMC) methods. It implements two sampling strategies:

- `Hydrogen MCMC`: A standard Metropolis-Hastings random walk.

- `Hydrogen MCMCMC`: A Metropolis-coupled MCMC (thermodynamic MCMC). Metropolis-coupled MCMCs run multiple MCMC chains in parallel at different temperatures. Only the coldest chain samples the target probability density, while hotter chains explore flattened versions of it. Periodic swaps between chains allow the cold chain to traverse low-probability barriers, greatly improving mixing.