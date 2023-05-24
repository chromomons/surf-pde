from ngsolve import SetNumThreads
import pandas as pd
import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
import seaborn as sns
import json
from math import pi

from utils import *
from laplace_solvers import diffusion
from exact import Exact

SetNumThreads(16)

f = open('input/input_fixed_surface_diffusion_test.json')
args = json.load(f)

mode = args['mode']
order = args['order']
alpha = args['alpha']
nu = args['nu']
tfinal = args['tfinal']
stab_type = args['stab_type']
unif_ref = args['unif_ref']
max_nref = args['max_nref']
fname = args['fname']
plt_out = args['plt_out']
bad_rhs = args['bad_rhs']
lamb = args['lamb']

# EXACT QUANTITIES

if mode == 'circ':
    R = 1.0
    params = {"nu": nu, "alpha": alpha, "R": R}
    exact = Exact(params)
    t = exact.t
    cfs = {
        "phi": sqrt(x*x + y*y + z*z) - R,
        "u": (1 + sin(pi * t)) * sin(pi * x) * sin(pi * y * z),
        "f": pi*(2*nu*x*(2*pi*y*z*cos(pi*y*z) + sin(pi*y*z))*(sin(pi*t) + 1)*cos(pi*x) + (6*nu*y*z*(sin(pi*t) + 1)*cos(pi*y*z) + (alpha*(x**2 + y**2 + z**2)*cos(pi*t) + pi*nu*(sin(pi*t) + 1)*(y**4 + y**2*(x**2 - 2*z**2 + 1) + z**2*(x**2 + z**2 + 1)))*sin(pi*y*z))*sin(pi*x))/(x**2 + y**2 + z**2)
    }
    exact.set_cfs(cfs)
    bbox_sz = 1.0
elif mode == 'circ-rough':
    R = 1.0
    params = {"nu": nu, "alpha": alpha, "params": lamb, "R": R}
    exact = Exact(params)
    t = exact.t
    tfun = 1 + sin(pi * t)
    cfs = {
        "phi": sqrt(x * x + y * y + z * z) - 1,
        "u": tfun * (sin(acos(z))) ** lamb * sin(atan2(y, x)),
        "f": tfun * ((lamb ** 2 + lamb) * (sin(acos(z))) ** lamb * sin(atan2(y, x)) + (1 - lamb ** 2) * (sin(acos(z))) ** (lamb - 2) * sin(atan2(y, x))) + pi*cos(pi*t) * (sin(acos(z))) ** lamb * sin(atan2(y, x)),
        "fel": alpha * (sin(acos(z))) ** lamb * sin(atan2(y, x)) + nu * ((lamb ** 2 + lamb) * (sin(acos(z))) ** lamb * sin(atan2(y, x)) + (1 - lamb ** 2) * (sin(acos(z))) ** (lamb - 2) * sin(atan2(y, x)))
    }
    exact.set_cfs(cfs)
    bbox_sz = 1.0
else:
    print("Invalid mode.")
    exit(1)

df = pd.DataFrame(columns=['h', 'dt', 'ndof', 'l2u', 'h1u'])

mesh = None

l2us = []
h1us = []

sns.set()

for nref in range(max_nref+1):
    exact.set_time(0.0)
    phi = exact.cfs['phi']

    h = 2*bbox_sz*2**(-unif_ref-nref)
    if bad_rhs:
        dt = h ** ((lamb + 1) / 2)
    else:
        dt = h**((order+1)/2)

    if mesh:
        refine_at_levelset(mesh=mesh, levelset=phi, nref=1)
    else:
        mesh = background_mesh(unif_ref=unif_ref, bbox_sz=bbox_sz)

    ndof, ts, l2uss, h1uss = diffusion(mesh=mesh, dt=dt, exact=exact,
                                       tfinal=tfinal, order=order, out=False, stab_type=stab_type)

    if plt_out:
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
    # df.to_csv(f"./csvs/diffusion/{mode}-p{order}-{stab_type}-{fname}.csv")

    if len(l2us) > 0:
        print(f"{ndof:.2E} & {np.log2(l2us[-1]/l2u):.2f} & {l2u:.2E} & {np.log2(h1us[-1]/h1u):.2f} & {h1u:.2E}")
    else:
        print(f"  ndof   &      &   lil2u  &      &   l2h1u")
        print(f"{ndof:.2E} &      & {l2u:.2E} &      & {h1u:.2E}")

    l2us.append(l2u)
    h1us.append(h1u)
