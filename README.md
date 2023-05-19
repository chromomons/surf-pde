# Surface PDE solvers
### Author: Mansur Shakipov
This repository hosts a collection of TraceFEM [1] routines that solve the following surface PDEs:
- Laplace-Beltrami equation
- Surface diffusion equation on a fixed and an evolving [2] surface
- Tangential Surface Stokes [3, 4] (steady and unsteady) equation on a fixed surface
- Tangential Surface Navier-Stokes (unsteady only) on a fixed and an evolving surfaces [5, 6]

This project is implemented in `ngsxfem` [7] add-on to `NGSolve` [8] library.

### References
[1] Maxim A. Olshanskii, Arnold Reusken, and J¨org Grande. “A Finite Element Method for Elliptic Equations on Surfaces”. en. In: SIAM Journal on Numerical Analysis 47.5 (Jan. 2009), pp. 3339–3358. issn: 0036-1429, 1095-7170. doi: 10.1137/080717602. url: http://epubs.siam.org/doi/10.1137/080717602. 

[2] Christoph Lehrenfeld, Maxim A. Olshanskii, and Xianmin Xu. A stabilized trace finite element method for partial differential equations on evolving surfaces. arXiv:1709.07117 [math]. Mar. 2018. url: http://arxiv.org/abs/1709.07117.

[3] Maxim A. Olshanskii et al. A finite element method for the surface Stokes problem. Number: arXiv:1801.06589 arXiv:1801.06589 [math]. Jan. 2018. url: http://arxiv.org/abs/1801.06589.

[4] Philip Brandner et al. Finite element discretization methods for velocity-pressure and stream function formulations of surface Stokes equations. arXiv:2103.03843 [cs, math]. Feb. 2022. url: http://arxiv.org/abs/2103.03843.

[5] Maxim A. Olshanskii, Arnold Reusken, and Alexander Zhiliakov. Tangential Navier-Stokes equations on evolving surfaces: Analysis and simulations. arXiv:2203.01521 [cs, math]. Mar. 2022. url: http://arxiv.org/abs/2203.01521.

[6] Maxim A. Olshanskii, Arnold Reusken, and Paul Schwering. An Eulerian finite element method  for tangential Navier-Stokes equations on evolving surfaces. arXiv:2302.00779 [math-ph]. Feb. 2023. url: http://arxiv.org/abs/2302.00779.

[7] Christoph Lehrenfeld et al. “ngsxfem: Add-on to NGSolve for geometrically unfitted finite element discretizations”. In: Journal of Open Source Software 6.64 (Aug. 2021), p. 3237. issn: 2475-9066. doi: 10.21105/joss.03237. url: https://joss.theoj.org/papers/10.21105/joss.03237.

[8] Joachim Schöberl. NGSolve Finite Element Library. Jan. 2009. url: http://sourceforge.net/projects/ngsolve.
