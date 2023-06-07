# Surface PDE solvers
### Author: Mansur Shakipov
This repository hosts a collection of TraceFEM [1] routines that solve the following surface PDEs:
- Laplace-Beltrami equation
- Surface diffusion equation on a fixed and an evolving [2] surface
- Tangential Surface Stokes [3, 4] (steady and unsteady) equation on a fixed surface
- Tangential Surface Navier-Stokes (unsteady only) on a fixed and an evolving surfaces [5, 6]

See [wiki](https://github.com/chromomons/surf-pde/wiki) for more details.

This project is implemented in `ngsxfem` [7] add-on to `NGSolve` [8] library.

### References
[1] Maxim A. Olshanskii, Arnold Reusken, and Jörg Grande. “A Finite Element Method for Elliptic Equations on Surfaces”. en. In: SIAM Journal on Numerical Analysis 47.5 (Jan. 2009), pp. 3339–3358. issn: 0036-1429, 1095-7170. doi: 10.1137/080717602. url: http://epubs.siam.org/doi/10.1137/080717602. 

[2] Christoph Lehrenfeld, Maxim A. Olshanskii, and Xianmin Xu. A stabilized trace finite element method for partial differential equations on evolving surfaces. arXiv:1709.07117 [math]. Mar. 2018. url: http://arxiv.org/abs/1709.07117.

[3] Maxim A. Olshanskii et al. A finite element method for the surface Stokes problem. Number: arXiv:1801.06589 arXiv:1801.06589 [math]. Jan. 2018. url: http://arxiv.org/abs/1801.06589.

[4] Philip Brandner et al. Finite element discretization methods for velocity-pressure and stream function formulations of surface Stokes equations. arXiv:2103.03843 [cs, math]. Feb. 2022. url: http://arxiv.org/abs/2103.03843.

[5] Maxim A. Olshanskii, Arnold Reusken, and Alexander Zhiliakov. Tangential Navier-Stokes equations on evolving surfaces: Analysis and simulations. arXiv:2203.01521 [cs, math]. Mar. 2022. url: http://arxiv.org/abs/2203.01521.

[6] Maxim A. Olshanskii, Arnold Reusken, and Paul Schwering. An Eulerian finite element method  for tangential Navier-Stokes equations on evolving surfaces. arXiv:2302.00779 [math-ph]. Feb. 2023. url: http://arxiv.org/abs/2302.00779.

[7] Christoph Lehrenfeld et al. “ngsxfem: Add-on to NGSolve for geometrically unfitted finite element discretizations”. In: Journal of Open Source Software 6.64 (Aug. 2021), p. 3237. issn: 2475-9066. doi: 10.21105/joss.03237. url: https://joss.theoj.org/papers/10.21105/joss.03237.

[8] Joachim Schöberl. NGSolve Finite Element Library. Jan. 2009. url: http://sourceforge.net/projects/ngsolve.

# Short manual

### My environment

Compilers and interpreters:
- `python=3.10.6`
- `gcc=11.3.0`
- `g++=11.3.0`

Libraries:
- `ngsolve=6.2.2105`
- `ngsxfem=2.0.2105`
- `numpy=1.23.0`
- `scipy=1.8.1`

My machine:
- `CPU: AMD Ryzen 7 5800H with Radeon Graphics`
- `RAM: 16GB`
- `Cores: 8`
- `Threads per core: 2`
- `Caches (sum of all)`:     
  - `L1d: 256 KiB (8 instances)`
  - `L1i: 256 KiB (8 instances)`
  - `L2:  4 MiB (8 instances)`
  - `L3:  16 MiB (1 instance)`

### Installation
Assuming `ngsxfem=2.0.2105` and `ngsolve=6.2.2105` are installed correcly, it should suffice to just clone this repository and run it.

### Files
Utility/helper files:
- `utils.py`: provides utility functions for both solvers and testers.
- `math2py.py`: converts Wolfram Mathematica symbolic expressions to python expressions.

Solvers:
- `laplace_solvers.py`: contains fixed-surface Poisson and diffusion, and evolving-surface diffusion solvers.
- `stokes_solvers.py`: contains fixed-surface steady and unsteady Stokes, fixed-surface unsteady Navier-Stokes, and evolving-surface Navier-Stokes solvers.

Testers:
- `fixed_surface_poisson_test.py`: tests fixed-surface Poisson solver.
- `fixed_surface_diffusion_test.py`: tests fixed-surface diffusion solver.
- `fixed_surface_steady_stokes_test.py`: tests steady Stokes solver.
- `fixed_surface_unsteady_stokes_test.py`: tests unsteady Stokes solver.
- `fixed_surface_navier_stokes_test.py`: tests fixed-surface Navier-Stokes.
- `evolving_surface_diffusion_test.py`: tests evolving-surface diffusion solver.
- `evolving_surface_navier_stokes_test.py`: tests evolving-surface Navier-Stokes

`input/` folder contains JSON input files for the testers.

`wolfram_mathamatica_notebooks/` folder contains Mathematica scripts that compute symbolic right-hand sides, and other exact quantities. 
