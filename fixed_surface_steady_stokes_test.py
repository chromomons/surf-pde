import numpy as np
from ngsolve import SetNumThreads
import pandas as pd
import json
import sys

from utils import math_dict_to_cfs, refine_at_levelset, background_mesh, printbf, print_test_info
from stokes_solvers import steady_stokes
from exact import Exact

num_cl_args = len(sys.argv)
if num_cl_args <= 1:
    print("Need to specify path to the JSON input file.")
    exit(1)
elif num_cl_args >= 3:
    print("Too many input arguments. Need to only specify path to the JSON input file.")
    exit(1)
else:
    input_file_name = str(sys.argv[1])

problem_name = "fixed_surface_steady_stokes"
print_test_info(problem_name)

f = open(input_file_name)

# loading JSON input files
args = json.load(f)

# Setting number of threads
SetNumThreads(args['num_threads'])

path_to_math_json = args['path_to_math_json']
f_math = open(path_to_math_json)
args_math = json.load(f_math)

# name of the test
name = args['name']

# orders: space order only in this case
orders = args['orders']
space_order = orders['space_order']

# PDE parameters
pde_params = args['pde_params']
# Solution parameters (can be useful if solution regularity depends on a parameter)
soln_params = args['soln_params']
# Parameters of the levelset
lset_params = args['lset_params']
# Meshing parameters
meshing_params = args['meshing_params']
# Output flags
out_params = args['out_params']
# Linear solver params
linear_solver_params = args['linear_solver_params']

# Unpacking some of the parameters

# text output of errors, separated by & for latex tables
txt_out = out_params['txt_out']
# vtk_out for VTK output
vtk_out = out_params['vtk_out']
# csv output of errors at each time t, useful with time-dependent problems
csv_out = out_params['csv_out']
# solver logs (i.e. how much time is spend in assembly and linear solver)
solver_logs = out_params['solver_logs']
# might be useful
print_sparse_solver_rates = out_params['print_sparse_solver_rates']

# half the size of the bounding box
bbox_sz = meshing_params['bbox_sz']
# number of uniform refinements
unif_ref = meshing_params['unif_ref']
# number of refinements at or around levelset
max_nref = meshing_params['max_nref']

# EXACT SOLUTION

# Collecting all parameters for coefficient functions
exact_params = {}

# add parameters of the PDE
for key, value in pde_params.items():
    exact_params[key] = value
# add parameters of the solution
for key, value in soln_params.items():
    exact_params[key] = value
# add parameters of the levelset
for key, value in lset_params.items():
    exact_params[key] = value

# Creating Exact object
exact = Exact(name=name, params=exact_params)
cfs = math_dict_to_cfs(args_math, exact_params)
exact.set_cfs(cfs)

df = pd.DataFrame(columns=['h', 'ndof', 'l2u', 'h1u', 'l2p', 'h1p'])

mesh = None

l2us, h1us, l2ps, h1ps = [], [], [], []

msg = f"&     h    &    dof    &  rate &    l2u    &  rate &    h1u    &  rate &    l2p    &  rate &    h1p    "
printbf(msg)

if txt_out:
    fe = open(f"./output/txt_out/{problem_name}_errs_p{space_order}_{name}.txt", "w")
    fe.write(f"{msg}\n")
    fe.close()

for nref in range(max_nref+1):
    h = bbox_sz * 2 ** (1 - unif_ref - nref)
    phi = exact.cfs['phi']

    if mesh:
        refine_at_levelset(mesh=mesh, levelset=phi, nref=1)
    else:
        mesh = background_mesh(unif_ref=unif_ref, bbox_sz=bbox_sz)

    vtk_out_postfix = f"h={h}" if vtk_out else None
    ndof, l2u, h1u, l2p, h1p = steady_stokes(
        mesh=mesh, exact=exact, order=space_order, vtk_out=vtk_out_postfix,
        logs=solver_logs, printrates=print_sparse_solver_rates, linear_solver_params=linear_solver_params
    )

    if len(l2us) > 0:
        msg = f"& $2^{{{int(np.log2(h))}}}$ & {ndof:.3E} & {np.log2(l2us[-1] / l2u):.3f} & {l2u:.3E} & {np.log2(h1us[-1] / h1u):.3f} & {h1u:.3E} & {np.log2(l2ps[-1] / l2p):.3f} & {l2p:.3E} & {np.log2(h1ps[-1] / h1p):.3f} & {h1p:.3E}"
    else:
        msg = f"& $2^{{{int(np.log2(h))}}}$ & {ndof:.3E} &       & {l2u:.3E} &       & {h1u:.3E} &       & {l2p:.3E} &       & {h1p:.3E}"

    l2us.append(l2u)
    h1us.append(h1u)
    l2ps.append(l2p)
    h1ps.append(h1p)

    # OUTPUT

    print(msg)

    if txt_out:
        fe = open(f"./output/txt_out/{problem_name}_errs_p{space_order}_{name}.txt", "a")
        fe.write(f"{msg}\n")
        fe.close()

    if csv_out:
        df.loc[nref] = [h, ndof, l2u, h1u]
        df.to_csv(f"./output/csv_out/{problem_name}_data_p{space_order}_{name}.csv")
