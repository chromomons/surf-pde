from laplace_solvers import *
from meshing import *
from ngsolve import SetNumThreads
import seaborn as sns

SetNumThreads(16)

unif_ref = 3
max_nref = 4

# EXACT QUANTITIES

exact = {
    "phi": sqrt(x*x + y*y + z*z) - 1
}
bbox_sz = 1.0

mesh = None

sns.set()

dts_o = [7.40E-06, 6.56E-07, 1.75E-08, 1.83E-09, 1.72E-10]
dts_n = [9.91E-07, 4.52E-07, 1.08E-08, 4.15E-09, 3.10E-10]
dts_t = [3.43E-06, 3.01E-07, 2.65E-08, 2.33E-09, 1.88E-10]

conds_o = []
conds_n = []
conds_t = []

for nref in range(max_nref+1):
    h = 2*bbox_sz*2**(-unif_ref-nref)

    dt_o = dts_o[nref]
    dt_n = dts_n[nref]
    dt_t = dts_t[nref]

    dts = {"old": dt_o,
           "new": dt_n,
           "total": dt_t}

    if mesh:
        refine_at_levelset(mesh=mesh, levelset=exact['phi'], nref=1)
    else:
        mesh = background_mesh(unif_ref=unif_ref, bbox_sz=bbox_sz)

    ndof, conds = compute_cond(mesh=mesh, dts=dts, ref_lvl=nref, **exact)
    cond_o = conds["old"]
    cond_n = conds["new"]
    cond_t = conds["total"]

    print(f"h = {h}, ndof = {ndof}")

    if nref > 0:
        print(f"{ndof:.2E} & {np.log2(dts_o[nref-1]/dt_o):.2f} & {dt_o:.2E} & {np.log2(cond_o/conds_o[-1]):.2f} & {cond_o:.2E} & {np.log2(dts_n[nref-1]/dt_n):.2f} & {dt_n:.2E} & {np.log2(cond_n/conds_n[-1]):.2f} & {cond_n:.2E} & {np.log2(dts_t[nref-1]/dt_t):.2f} & {dt_t:.2E} & {np.log2(cond_t/conds_t[-1]):.2f} & {cond_t:.2E}")
    else:
        print(f"  ndof   &      &   dt_o   &      &  cond_o  &      &   dt_n   &      &  cond_n  &      &   dt_t   &      &  cond_t  ")
        print(f"{ndof:.2E} &      & {dt_o:.2E} &      & {cond_o:.2E} &      & {dt_n:.2E} &      & {cond_n:.2E} &      & {dt_t:.2E} &      & {cond_t:.2E}")

    conds_o.append(cond_o)
    conds_n.append(cond_n)
    conds_t.append(cond_t)
