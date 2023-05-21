import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from laplace_solvers import *
import numpy as np
from utils import *
from ngsolve import SetNumThreads
import pandas as pd
import sys
from math import pi

SetNumThreads(16)

mode = str(sys.argv[1])
order = int(sys.argv[2])

mass_cf = 1

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
    max_nref = 5
else:
    print("Invalid mode.")
    exit(1)

df = pd.DataFrame(columns=['h', 'ndof', 'l2u', 'h1u'])

mesh = None

l2us = []
h1us = []
l2ps = []
h1ps = []

for nref in range(max_nref+1):
    h = 2*bbox_sz*2**(-unif_ref-nref)

    if mesh:
        refine_at_levelset(mesh=mesh, levelset=exact['phi'], nref=1)
    else:
        mesh = background_mesh(unif_ref=unif_ref, bbox_sz=bbox_sz)

    ndof, l2u, h1u = poisson(mesh=mesh, order=order, mass_cf=mass_cf, out=False, **exact)

    print(f"h = {h}, ndof = {ndof}")

    df.loc[nref] = [h, ndof, l2u, h1u]
    # df.to_csv(f"./csvs/poisson_{mode}_p{order}.csv")

    if len(l2us) > 0:
        print(f"{ndof:.2E} & {np.log2(l2us[-1]/l2u):.2f} & {l2u:.2E} & {np.log2(h1us[-1]/h1u):.2f} & {h1u:.2E}")
    else:
        print(f"{ndof:.2E} &      & {l2u:.2E} &      & {h1u:.2E}")

    l2us.append(l2u)
    h1us.append(h1u)
