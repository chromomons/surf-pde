import numpy as np
from ngsolve import SetNumThreads
import pandas as pd
import json
from math import pi

from utils import *
from laplace_solvers import poisson
from exact import Exact

SetNumThreads(16)

f = open('input/input_fixed_surface_poisson_test.json')
args = json.load(f)

mode = args['mode']
order = args['order']
alpha = args['alpha']
nu = args['nu']
unif_ref = args['unif_ref']
max_nref = args['max_nref']
fname = args['fname']
plt_out = args['plt_out']
lamb = args['lamb']

# EXACT QUANTITIES

if mode == 'circ':
    R = 1.0
    params = {"nu": nu, "alpha": alpha, "R": R}
    exact = Exact(params)
    cfs = {
        "phi": sqrt(x*x + y*y + z*z) - R,
        "u": sin(pi * x) * sin(pi * y * z),
        "f": (2*pi*nu*x*(2*pi*y*z*cos(pi*y*z) + sin(pi*y*z))*cos(pi*x) + (6*pi*nu*y*z*cos(pi*y*z) + (alpha*(x**2 + y**2 + z**2) + pi**2*nu*(y**4 + y**2*(x**2 - 2*z**2 + 1) + z**2*(x**2 + z**2 + 1)))*sin(pi*y*z))*sin(pi*x))/(x**2 + y**2 + z**2)
    }
    exact.set_cfs(cfs)
    bbox_sz = 1.0
elif mode == 'circ-rough':
    R = 1.0
    params = {"nu": nu, "alpha": alpha, "params": lamb, "R": R}
    exact = Exact(params)
    cfs = {
        "phi": sqrt(x * x + y * y + z * z) - R,
        "u": (sin(acos(z))) ** lamb * sin(atan2(y, x)),
        "f": nu * ((lamb ** 2 + lamb) * (sin(acos(z))) ** lamb * sin(atan2(y, x)) + (1 - lamb ** 2) * (sin(acos(z))) ** (lamb - 2) * sin(atan2(y, x))) + alpha * (sin(acos(z))) ** lamb * sin(atan2(y, x))
    }
    exact.set_cfs(cfs)
    bbox_sz = 1.0
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
    phi = exact.cfs['phi']

    if mesh:
        refine_at_levelset(mesh=mesh, levelset=phi, nref=1)
    else:
        mesh = background_mesh(unif_ref=unif_ref, bbox_sz=bbox_sz)

    ndof, l2u, h1u = poisson(mesh=mesh, exact=exact, order=order, out=False)

    print(f"h = {h}, ndof = {ndof}")

    df.loc[nref] = [h, ndof, l2u, h1u]
    # df.to_csv(f"./csvs/poisson_{mode}_p{order}.csv")

    if len(l2us) > 0:
        print(f"{ndof:.2E} & {np.log2(l2us[-1]/l2u):.2f} & {l2u:.2E} & {np.log2(h1us[-1]/h1u):.2f} & {h1u:.2E}")
    else:
        print(f"{ndof:.2E} &      & {l2u:.2E} &      & {h1u:.2E}")

    l2us.append(l2u)
    h1us.append(h1u)
