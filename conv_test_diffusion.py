from laplace_solvers import *
from meshing import *
from ngsolve import SetNumThreads
import pandas as pd
import numpy as np
import scipy.integrate as sci
import sys
import matplotlib.pyplot as plt


SetNumThreads(16)

mode = str(sys.argv[1])
order = int(sys.argv[2])

mass_cf = 0.0

# EXACT QUANTITIES

if mode == 'circ':
    exact = {
        "name": mode,
        "phi": sqrt(x*x + y*y + z*z) - 1,
        "u": sin(pi * x) * sin(pi * y * z),
        "f": pi*(2*y*z*(2*pi*x*cos(pi*x) + 3*sin(pi*x))*cos(pi*y*z) + (2*x*cos(pi*x) + pi*(y**4 + y**2*(x**2 - 2*z**2 + 1) + z**2*(x**2 + z**2 + 1))*sin(pi*x))*sin(pi*y*z))/(x**2 + y**2 + z**2) + mass_cf*sin(pi*x)*sin(pi*y*z)
    }
    bbox_sz = 1.0
    unif_ref = 2
    max_nref = 4
else:
    print("Invalid mode.")
    exit(1)

df = pd.DataFrame(columns=['h', 'dt', 'ndof', 'l2u', 'h1u'])

mesh = None

l2us = []
h1us = []

for nref in range(max_nref+1):
    h = 2*bbox_sz*2**(-unif_ref-nref)
    dt = h**1.5

    if mesh:
        refine_at_levelset(mesh=mesh, levelset=exact['phi'], nref=1)
    else:
        mesh = background_mesh(unif_ref=unif_ref, bbox_sz=bbox_sz)

    ndof, ts, l2uss, h1uss = diffusion(mesh=mesh, dt=dt, order=order, out=False, **exact)

    plt.plot(l2uss)
    plt.title('L^2 vels')
    plt.show()

    plt.plot(h1uss)
    plt.title('H^1 vels')
    plt.show()


    l2u = max(l2uss)
    h1u = np.sqrt(sci.simps(y=np.array(h1uss)**2, x=ts, dx=dt, even='avg'))

    print(f"h = {h}, ndof = {ndof}")

    df.loc[nref] = [h, dt, ndof, l2u, h1u]
    # df.to_csv(f"./csvs/diffusion_{mode}_p{order}_old_stab.csv")

    if len(l2us) > 0:
        print(f"{ndof:.2E} & {np.log2(l2us[-1]/l2u):.2f} & {l2u:.2E} & {np.log2(h1us[-1]/h1u):.2f} & {h1u:.2E}")
    else:
        print(f"  ndof   &      &   lil2u  &      &   l2h1u")
        print(f"{ndof:.2E} &      & {l2u:.2E} &      & {h1u:.2E}")

    l2us.append(l2u)
    h1us.append(h1u)
