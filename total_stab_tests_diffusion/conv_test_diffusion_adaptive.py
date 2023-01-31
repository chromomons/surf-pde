from laplace_solvers import *
from meshing import *
from ngsolve import SetNumThreads
import pandas as pd
import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
import seaborn as sns
import json

SetNumThreads(16)

f = open('input_diffusion_adaptive.json')
params = json.load(f)

mode = params['mode']
order = params['order']
stab_type = params['stab_type']
unif_ref = params['unif_ref']
max_nref = params['max_nref']
fname = params['fname']
plt_out = params['plt_out']
bad_rhs = params['bad_rhs']

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
elif mode == 'circ-rough':
    lamb = params['lamb']
    exact = {
        "name": mode,
        "phi": sqrt(x * x + y * y + z * z) - 1,
        "u": (sin(acos(z))) ** lamb * sin(atan2(y, x)),
        "f": (lamb ** 2 + lamb) * (sin(acos(z))) ** lamb * sin(atan2(y, x)) + (1 - lamb ** 2) * (sin(acos(z))) ** (lamb - 2) * sin(atan2(y, x))
    }
    bbox_sz = 1.0
else:
    print("Invalid mode.")
    exit(1)

df = pd.DataFrame(columns=['h', 'dt', 'ndof', 'l2u', 'h1u'])

mesh = None

l2us = []
h1us = []
cg_sums = []
cg_means = []
dt_mins = []

sns.set()

fe = open(f"{stab_type}-errs.txt", "w")
fd = open(f"{stab_type}-data.txt", "w")

for nref in range(max_nref+1):
    h = 2*bbox_sz*2**(-unif_ref-nref)

    if mesh:
        refine_at_levelset(mesh=mesh, levelset=exact['phi'], nref=1)
    else:
        mesh = background_mesh(unif_ref=unif_ref, bbox_sz=bbox_sz)

    ndof, ts, l2uss, h1uss, dts, cg_iters = diffusion_adaptive(mesh=mesh, order=order, out=False, stab_type=stab_type, bad_rhs=bad_rhs, **exact)

    fd.write(str(ts[1:]))
    fd.write("\n")
    fd.write(str(cg_iters))
    fd.write("\n")

    if plt_out:
        plt.plot(ts[1:], cg_iters)
        ax = plt.gca()
        ax.set_ylim([0, None])
        plt.title('CG iters')
        plt.show()

    l2u = max(l2uss)
    h1u = np.sqrt(sci.simps(y=np.array(h1uss)**2, x=ts, dx=dt, even='avg'))
    cg_sum = np.sum(cg_iters)
    cg_mean = np.mean(cg_iters)
    dt_min = np.min(dts)

    print(f"h = {h}, ndof = {ndof}")

    df.loc[nref] = [h, dt, ndof, l2u, h1u]
    df.to_csv(f"./csvs/diffusion/adaptive-{mode}-p{order}-{stab_type}-{fname}.csv")

    if len(l2us) > 0:
        print(f"{ndof:.2E} & {np.log2(l2us[-1] / l2u):.2f} & {l2u:.2E} & {np.log2(h1us[-1] / h1u):.2f} & {h1u:.2E} & {np.log2(cg_sum / cg_sums[-1]):.2f} & {cg_sum:.2E} & {np.log2(cg_mean / cg_means[-1]):.2f} & {cg_mean:.2E} & {np.log2(dt_mins[-1] / dt_min):.2f} & {dt_min:.2E}")
        fe.write(f"{ndof:.2E} & {np.log2(l2us[-1]/l2u):.2f} & {l2u:.2E} & {np.log2(h1us[-1]/h1u):.2f} & {h1u:.2E} & {np.log2(cg_sum/cg_sums[-1]):.2f} & {cg_sum:.2E} & {np.log2(cg_mean/cg_means[-1]):.2f} & {cg_mean:.2E} & {np.log2(dt_mins[-1]/dt_min):.2f} & {dt_min:.2E}")
        fe.write("\n")
    else:
        print(f"  ndof   &      &   lil2u  &      &   l2h1u  &      &  cgsum   &      &  cgmean  &      &  dt_min  ")
        print(f"{ndof:.2E} &      & {l2u:.2E} &      & {h1u:.2E} &      & {cg_sum:.2E} &      & {cg_mean:.2E} &      & {dt_min:.2E}")
        fe.write(f"  ndof   &      &   lil2u  &      &   l2h1u  &      &  cgsum   &      &  cgmean  &      &  dt_min  ")
        fe.write("\n")
        fe.write(f"{ndof:.2E} &      & {l2u:.2E} &      & {h1u:.2E} &      & {cg_sum:.2E} &      & {cg_mean:.2E} &      & {dt_min:.2E}")
        fe.write("\n")

    l2us.append(l2u)
    h1us.append(h1u)
    cg_sums.append(cg_sum)
    cg_means.append(cg_mean)
    dt_mins.append(dt_min)

fe.close()
fd.close()
