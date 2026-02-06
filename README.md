# (Metropolis-Coupled) Monte Carlo Markov Chain

This project visualizes the electron orbitals of a hydrogen atom by sampling the wavefunction probability density $|\psi_{n \ell m}(r, \theta, \varphi)|^2$ using Markov Chain Monte Carlo (MCMC) methods. It implements two sampling strategies:

- `Hydrogen MCMC`: A standard Metropolis-Hastings random walk.

- `Hydrogen MCMCMC`: A [Metropolis-coupled MCMC](http://bamm-project.org/mc3.html) (thermodynamic MCMC). Metropolis-coupled MCMCs run multiple MCMC chains in parallel at different temperatures. Only the coldest chain samples the target probability density, while hotter chains explore flattened versions of it. Periodic swaps between chains allow the cold chain to traverse low-probability barriers, greatly improving mixing.

![](https://github.com/epa058/MC-MCMC/blob/main//Hydrogen%20MCMCMC.gif)

## FAQ

#### Q: Why did you start this project?

A: Because I didn't know what MCMCs were, and that was making me insecure. Actually, that's not even true. I've used the Metropolis algorithm before to simulate Ising models. Sorry Adrian, I promise I learned a lot from your class.

#### Q: What if I want 2D visualizations?

A: Go [here](https://github.com/ssebastianmag/hydrogen-wavefunctions). 

#### Q: What next?

A: The actual reason for wanting to do this project was because I came across [this video by minutephysics](https://www.youtube.com/watch?v=W2Xb2GFK2yc) a few years ago and wanted to recreate their animations, but couldn't find any of their source files. Turns out someone else had the exact same idea and beat me to it: [https://asliceofcuriosity.fr/blog/posts/rendering3.html](https://asliceofcuriosity.fr/blog/posts/rendering3.html) (Wagyx). Like them, I also want to get better at JavaScript, so I will try to make my own application. However, unlike them, I am lazy and will probably not get to it any time soon.
