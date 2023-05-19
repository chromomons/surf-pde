# ------------------------------ LOAD LIBRARIES -------------------------------
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from xfem.lsetcurv import *
from ngsolve import solvers
from ngsolve import TaskManager
import time
import sys
import numpy as np
from math import pi
from netgen.csg import CSGeometry, OrthoBrick, Pnt
import matplotlib.pyplot as plt
import scipy.integrate as sci
import seaborn as sns
from netgen.geom2d import SplineGeometry

# FORMATTING TOOLS
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printbf(s):
    print(f"{bcolors.BOLD}{s}{bcolors.ENDC}")
    return


# DIFF OPERATORS


def coef_fun_grad(u):
    return CoefficientFunction(tuple([u.Diff(d) for d in [x,y,z]]))


def vec_grad(v):
    return CoefficientFunction(tuple( [v[i].Diff(d) for i in [0,1,2] for d in [x,y,z]] ), dims=(3,3))


# ERRORS
def get_dom_measure(Q, mesh, ds):
    one = GridFunction(Q)
    one.Set(CoefficientFunction(1.0))
    domMeas = Integrate(one * ds, mesh)
    return domMeas


def renormalize(Q, mesh, ds, gfp, domMeas=None):
    gfpInt = Integrate(gfp * ds, mesh)
    if not domMeas:
        domMeas = get_dom_measure(Q, mesh, ds)
    gfpMeanVal = GridFunction(Q)
    gfpMeanVal.Set(CoefficientFunction(float(gfpInt/domMeas)))
    pNum = GridFunction(Q)
    pNum.Set(gfp - gfpMeanVal)
    gfp.Set(pNum)
    return


def errors(mesh, ds, Pmat, gfu, gfp, uSol, pSol):
    return sqrt(Integrate(InnerProduct(gfu - uSol, Pmat * (gfu - uSol)) * ds, mesh)),\
           sqrt(Integrate(InnerProduct((grad(gfu) - vec_grad(uSol)) * Pmat, Pmat * (grad(gfu) - vec_grad(uSol)) * Pmat) * ds, mesh)),\
           sqrt(Integrate((pSol - gfp) * (pSol - gfp) * ds, mesh)),\
           sqrt(Integrate(InnerProduct(grad(gfp) - coef_fun_grad(pSol), Pmat*(grad(gfp) - coef_fun_grad(pSol))) * ds, mesh))


def errors_u(mesh, ds, Pmat, gfu, uSol):
    return sqrt(Integrate(InnerProduct(Pmat * (gfu - uSol), Pmat * (gfu - uSol)) * ds, mesh)),\
           sqrt(Integrate(InnerProduct((grad(gfu) - vec_grad(uSol)) * Pmat, Pmat * (grad(gfu) - vec_grad(uSol)) * Pmat) * ds, mesh))


def errors_u_nontang(mesh, ds, Pmat, gfu, uSol):
    return sqrt(Integrate(InnerProduct((gfu - uSol), (gfu - uSol)) * ds, mesh)),\
           sqrt(Integrate(InnerProduct(Pmat * (grad(gfu) - vec_grad(uSol)) * Pmat, Pmat * (grad(gfu) - vec_grad(uSol)) * Pmat) * ds, mesh))

# HELPERS

def define_forms(eq_type, V, n, Pmat, rhsf, ds, dX, **args):
    u, v = V.TnT()
    h = specialcf.mesh_size

    if eq_type == 'poisson':
        # penalization parameters
        rho_u = 1.0 / h
        mass_cf = args['mass_cf']

        # a_h part
        a = BilinearForm(V, symmetric=True)
        a += mass_cf * u * v * ds
        a += InnerProduct(Pmat * grad(u), Pmat * grad(v)) * ds
        # normal gradient volume stabilization of the velocity
        a += rho_u * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX

        f = LinearForm(V)
        f += rhsf * v * ds

        return a, f
    else:
        # penalization parameters
        rho_u = 1.0 / h
        alpha = args['alpha']
        dt = args['dt']
        stab_type = args['stab_type']

        m = BilinearForm(V, symmetric=True)  # mass
        d = BilinearForm(V, symmetric=True)  # total_stab_tests_diffusion
        a = BilinearForm(V, symmetric=True)  # mass-total_stab_tests_diffusion

        # mass part
        m += u * v * ds
        a += 2.0 / (alpha * dt) * u * v * ds

        if stab_type in ['new', 'total']:
            # stabilizing mass part
            m += h * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX
            a += 2.0 / (alpha * dt) * h * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX

        if stab_type in ['old', 'total']:
            # stabilizing total_stab_tests_diffusion part
            d += rho_u * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX
            a += rho_u * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX

        # total_stab_tests_diffusion part
        a += InnerProduct(Pmat * grad(u), Pmat * grad(v)) * ds
        d += InnerProduct(Pmat * grad(u), Pmat * grad(v)) * ds

        f = LinearForm(V)
        f += rhsf * v * ds

        return m, d, a, f


def assemble_forms(list_of_forms):
    for form in list_of_forms:
        form.Assemble()


def append_errors(t_curr, l2u, h1u, l2p, h1p, **errs):
    errs['ts'].append(t_curr)
    errs['l2us'].append(l2u)
    errs['h1us'].append(h1u)
    errs['l2ps'].append(l2p)
    errs['h1ps'].append(h1p)


# SOLVERS
def background_mesh(bbox_sz):
    geo = CSGeometry()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                geo.Add(OrthoBrick(Pnt(min((-1) ** i * bbox_sz, 0),
                                       min((-1) ** j * bbox_sz, 0),
                                       min((-1) ** k * bbox_sz, 0)),
                                   Pnt(max((-1) ** i * bbox_sz, 0),
                                       max((-1) ** j * bbox_sz, 0),
                                       max((-1) ** k * bbox_sz, 0))))
    mesh = Mesh(geo.GenerateMesh(maxh=bbox_sz, quad_dominated=False))
    with TaskManager():
        mesh.Refine()
    return mesh


def refine_around_lset(mesh, unif_ref, phi, vel, c_delta, tfinal, dt, band_type='both'):
    for i in range(unif_ref - 2):
        lset_p1 = GridFunction(H1(mesh=mesh, order=1, dirichlet=[]))
        InterpolateToP1(phi, lset_p1)
        if band_type == 'outer':
            RefineAtLevelSet(lset_p1, lower=-vel * (c_delta * dt), upper=vel * (tfinal + c_delta * dt))
        elif band_type == 'inner':
            RefineAtLevelSet(lset_p1, lower=-vel * (tfinal + c_delta * dt), upper=vel * (c_delta * dt))
        else:
            RefineAtLevelSet(lset_p1, lower=-vel*(tfinal+c_delta*dt), upper=vel*(tfinal+c_delta*dt))
        with TaskManager():
            mesh.Refine()
    return


def update_ba_IF_band(lset_approx, mesh, band_size, ba_IF_band):
    VGrid = H1(mesh, order=1)
    lset_plus = GridFunction(VGrid)
    lset_minus = GridFunction(VGrid)
    InterpolateToP1(lset_approx - band_size, lset_plus)
    InterpolateToP1(lset_approx + band_size, lset_minus)
    ci_plus = CutInfo(mesh, lset_plus)
    ci_minus = CutInfo(mesh, lset_minus)

    ba_IF_band.Clear()
    ba_IF_band |= ci_plus.GetElementsOfType(HASNEG)
    ba_IF_band &= ci_minus.GetElementsOfType(HASPOS)
    return


def helper_grid_functions(mesh, order, levelset, vel_space):
    # aux pressure space of order=order+1
    Phkp1 = H1(mesh, order=order + 1, dirichlet=[])
    # aux pressure space of order=order
    Phk = H1(mesh, order=order, dirichlet=[])
    # aux velocity space of order=order-1 for the shape operator
    VPhkm1 = VectorH1(mesh, order=order - 1, dirichlet=[])

    phi_kp1 = GridFunction(Phkp1)
    n_k = GridFunction(vel_space)
    phi_k = GridFunction(Phk)
    n_km1 = GridFunction(VPhkm1)

    phi_kp1.Set(levelset)
    n_k.Set(Normalize(grad(phi_kp1)))
    phi_k.Set(levelset)
    n_km1.Set(Normalize(grad(phi_k)))
    Hmat = grad(n_km1)

    return n_k, Hmat


def moving_diffusion(order, unif_ref, bbox_sz, tfinal, exact_sol_type="translation", time_order=1, out=False):
    bdf_coeffs = [[1, -1],
                  [3 / 2, -2, 1 / 2],
                  [11 / 6, -3, 3 / 2, -1 / 3]]

    bdf_coeff = bdf_coeffs[time_order-1]

    c_delta = time_order + 0.5
    # MESH
    mesh = background_mesh(bbox_sz)

    h_approx = bbox_sz * 2 ** (1 - unif_ref)

    t = Parameter(0.0)

    if exact_sol_type == 'translation':
        dt = h_approx ** ((order + 1) / time_order) / 4

        vel = 0.2
        w1 = CoefficientFunction(vel)
        w2 = CoefficientFunction(0.0)
        w3 = CoefficientFunction(0.0)
        w = CoefficientFunction((w1, w2, w3))

        R = 1.0
        nu = R * R
        phi = sqrt((x - vel * t) * (x - vel * t) + y * y + z * z) - R
    else:
        dt = h_approx ** ((order + 1) / time_order) / 4

        vel = 1.0
        w1 = CoefficientFunction(-x*(t*(y**2 + z**2) + x**2)*(x**2*(y**2 + z**2) + y**2*z**2)/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)))
        w2 = CoefficientFunction(-y*(t*(x**2 + z**2) + y**2)*(x**2*(y**2 + z**2) + y**2*z**2)/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)))
        w3 = -z*(t*(x**2 + y**2) + z**2)*(x**2*(y**2 + z**2) + y**2*z**2)/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6))
        w = CoefficientFunction((w1, w2, w3))

        R = 1
        nu = 1

        phi = -R + (2*t*(x**2*y**2 + x**2*z**2 + y**2*z**2) + x**4 + y**4 + z**4)**(1/4)

    if exact_sol_type == 'translation':
        refine_around_lset(mesh, unif_ref, phi, vel, c_delta, tfinal, dt)

        coef_u = 1 + (x + y + z - vel * t) * exp(-2 * t)
        coef_f = 0.0
        divGwT = -2 * vel / (R * R) * (x - vel * t)
        divGw = 0.0
        coef_f_el = (-t**3*w1*(w1**2 + w2**2 + w3**2) + t**2*(w1**2*(3*x + y + z) + 2*w1*(w2*y + w3*z) + (w2**2 + w3**2)*(x + y + z) + (w1**2 + w2**2 + w3**2)*exp(2*t)) - 2*t*(w1*x + w2*y + w3*z)*exp(2*t) - t*(w1*(3*x**2 + 2*x*(y + z) + y**2 + z**2 + 2) + 2*w2 + 2*w3 + (w2*y + w3*z)*(2*x + 2*y + 2*z)) + 2*x + 2*y + 2*z + (x**2 + y**2 + z**2)*(x + y + z + exp(2*t)))*exp(-2*t)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)
    else:
        refine_around_lset(mesh, unif_ref, phi, vel, c_delta, tfinal, dt, band_type='inner')
        coef_u = (sin(pi*t) + 1)*sin(pi*x)
        coef_f = (pi*x*(2*nu*(t**3*x**6*y**2 + t**3*x**2*y**6 + 3*t**2*x**6*y**2 + 4*t**2*x**4*y**4 + 3*t**2*x**2*y**6 + 2*t*x**8 + t*x**6*y**2 + 8*t*x**4*y**4 + t*x**2*y**6 + 2*t*y**8 + 2*t*z**8 + 2*t*z**4*(x**4*(2*t + 4) + x**2*y**2*(-t*(t - 7) + 6) + y**4*(2*t + 4)) + 3*x**6*y**2 + 3*x**2*y**6 + z**6*(t + 3)*(t**2 + 1)*(x**2 + y**2) + z**2*(x**2 + y**2)*(x**4*(t + 3)*(t**2 + 1) - x**2*y**2*(t + 1)*(t*(3*t - 14) + 3) + y**4*(t + 3)*(t**2 + 1))) - (x**2*y**2 + z**2*(x**2 + y**2))*(t*x**4*y**2*(t + 2) + t*x**2*y**4*(t + 2) + t*z**4*(t + 2)*(x**2 + y**2) + t*z**2*(6*t*x**2*y**2 + x**4*(t + 2) + y**4*(t + 2)) + x**6 + y**6 + z**6))*(t*(y**2 + z**2) + x**2)*(sin(pi*t) + 1)*cos(pi*x) + ((sin(pi*t) + 1)*(2*pi**2*nu*t**4*x**8*y**4 + 2*pi**2*nu*t**4*x**6*y**6 + 4*pi**2*nu*t**3*x**8*y**4 + 8*pi**2*nu*t**3*x**6*y**6 + 4*pi**2*nu*t**3*x**4*y**8 + 2*pi**2*nu*t**2*x**10*y**2 + 8*pi**2*nu*t**2*x**6*y**6 + 12*pi**2*nu*t**2*x**4*y**8 + 2*pi**2*nu*t**2*x**2*y**10 + 4*pi**2*nu*t*x**8*y**4 + 4*pi**2*nu*t*x**4*y**8 + 8*pi**2*nu*t*x**2*y**10 + 2*pi**2*nu*x**6*y**6 + 2*pi**2*nu*y**12 + 2*pi**2*nu*z**12 - t**3*x**8*y**4 - t**3*x**4*y**8 - 3*t**2*x**8*y**4 - 4*t**2*x**6*y**6 - 3*t**2*x**4*y**8 - 2*t*x**10*y**2 - t*x**8*y**4 - 8*t*x**6*y**6 - t*x**4*y**8 - 2*t*x**2*y**10 + 2*t*z**10*(x**2*(pi**2*nu*(t + 4) - 1) + y**2*(2*pi**2*nu*(t + 2) - 1)) - 3*x**8*y**4 - 3*x**4*y**8 + z**8*(x**4*(t*(4*pi**2*nu - 1)*(t*(t + 3) + 1) - 3) + x**2*y**2*(2*t*(t*(pi**2*nu*(t*(t + 6) + 18) - t - 3) - 2) - 6) + y**4*(t*(2*pi**2*nu*(t + 2)*(t*(t + 2) + 2) - t*(t + 3) - 1) - 3)) + z**6*(x**6*(2*pi**2*nu*(t**2*(t + 2)**2 + 1) - 4*t*(t + 2)) + x**4*y**2*(t*(4*pi**2*nu*(t*(t*(3*t + 13) + 3) + 1) + t*(t - 21) - 21) - 3) + x**2*y**4*(t*(2*pi**2*nu*(t*(t*(11*t + 26) + 9) + 4) + t*(t - 21) - 21) - 3) + y**6*(4*pi**2*nu*(t**2*(t + 2)**2 + 1) - 4*t*(t + 2))) + z**4*(2*t*x**4*y**4*(4*pi**2*nu*t*(t*(7*t + 4) + 4) + t*(3*t - 21) - 18) + x**8*(t*(2*pi**2*nu*(t**2*(t + 2) + 2) - t*(t + 3) - 1) - 3) + x**6*y**2*(t*(2*pi**2*nu*(t*(t*(11*t + 12) + 5) + 2) + t*(t - 21) - 21) - 3) + x**2*y**6*(t*(2*pi**2*nu*(t*(t*(11*t + 26) + 9) + 4) + t*(t - 21) - 21) - 3) + y**8*(t*(2*pi**2*nu*(t + 2)*(t*(t + 2) + 2) - t*(t + 3) - 1) - 3)) + z**2*(2*t*x**10*(pi**2*nu*t - 1) + 2*t*y**10*(2*pi**2*nu*(t + 2) - 1) + x**8*y**2*(2*t*(t*(2*pi**2*nu*(t*(t + 2) + 2) - t - 3) - 2) - 6) + x**6*y**4*(t*(2*pi**2*nu*(t*(t*(11*t + 12) + 5) + 2) + t*(t - 21) - 21) - 3) + x**4*y**6*(t*(4*pi**2*nu*(t*(t*(3*t + 13) + 3) + 1) + t*(t - 21) - 21) - 3) + x**2*y**8*(2*t*(t*(pi**2*nu*(t*(t + 6) + 18) - t - 3) - 2) - 6))) + 2*pi*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**2*cos(pi*t))*sin(pi*x))/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**2)
        divGwT = 0
        divGw = -(x**2*(y**2 + z**2) + y**2*z**2)*(t**3*x**6*y**2 + t**3*x**2*y**6 + 3*t**2*x**6*y**2 + 4*t**2*x**4*y**4 + 3*t**2*x**2*y**6 + 2*t*x**8 + t*x**6*y**2 + 8*t*x**4*y**4 + t*x**2*y**6 + 2*t*y**8 + 2*t*z**8 + 2*t*z**4*(x**4*(2*t + 4) + x**2*y**2*(-t*(t - 7) + 6) + y**4*(2*t + 4)) + 3*x**6*y**2 + 3*x**2*y**6 + z**6*(t + 3)*(t**2 + 1)*(x**2 + y**2) + z**2*(x**2 + y**2)*(x**4*(t + 3)*(t**2 + 1) - x**2*y**2*(t + 1)*(t*(3*t - 14) + 3) + y**4*(t + 3)*(t**2 + 1)))/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**2)
        coef_f_el = (pi*x*(t*(y**2 + z**2) + x**2)*(t**3*x**6*y**2 + t**3*x**2*y**6 + 3*t**2*x**6*y**2 + 4*t**2*x**4*y**4 + 3*t**2*x**2*y**6 + 2*t*x**8 + t*x**6*y**2 + 8*t*x**4*y**4 + t*x**2*y**6 + 2*t*y**8 + 2*t*z**8 + 2*t*z**4*(x**4*(2*t + 4) + x**2*y**2*(-t*(t - 7) + 6) + y**4*(2*t + 4)) + 3*x**6*y**2 + 3*x**2*y**6 + z**6*(t + 3)*(t**2 + 1)*(x**2 + y**2) + z**2*(x**2 + y**2)*(x**4*(t + 3)*(t**2 + 1) - x**2*y**2*(t + 1)*(t*(3*t - 14) + 3) + y**4*(t + 3)*(t**2 + 1)))*cos(pi*x) + (t*x**4*(y**2 + z**2)*(t + pi**2*t + 2) + t*x**2*(t*y**2*z**2*(6 + 4*pi**2) + y**4*(t + 2 + 2*pi**2) + z**4*(t + 2 + 2*pi**2)) + x**6 + (1 + pi**2)*(y**2 + z**2)*(y**4 + y**2*z**2*(t*(t + 2) - 1) + z**4))*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)*sin(pi*x))*(sin(pi*t) + 1)/(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**2

    V = H1(mesh, order=order, dirichlet=[])

    # LEVELSET ADAPTATION
    lset_approx = GridFunction(H1(mesh, order=1, dirichlet=[]))
    InterpolateToP1(phi, lset_approx)
    ci = CutInfo(mesh, lset_approx)

    ba_IF = ci.GetElementsOfType(IF)
    ba_IF_band = BitArray(mesh.ne)
    update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

    # define projection matrix
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ba_IF)
    dX = dx(definedonelements=ba_IF_band)

    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)
    h = specialcf.mesh_size
    rho = 1./h/100

    gfu_prevs = []

    if out:
        gfu_out = GridFunction(V)
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, gfu_out, coef_u],
                        names=["P1-levelset", "u", "uSol"],
                        filename=f"./vtk_out/diffusion/moving-diff-new",
                        subdivision=0)

    out_errs = {'ts': [], 'l2us': [], 'h1us': []}

    for j in range(time_order):
        # fix levelset
        t.Set(-j*dt)
        t_curr = -j * dt

        # solve elliptic problem on a fixed surface to get u with normal extension
        gfu_temp = GridFunction(V)

        InterpolateToP1(phi, lset_approx)
        ci = CutInfo(mesh, lset_approx)

        ba_IF.Clear()
        ba_IF |= ci.GetElementsOfType(IF)
        update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

        VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
        gfu_el = GridFunction(VG)
        u_el, v_el = VG.TnT()

        a_el = BilinearForm(VG, symmetric=True)
        a_el += (u_el*v_el + InnerProduct(Pmat * grad(u_el), Pmat * grad(v_el))) * ds
        a_el += 1./h * (n * grad(u_el)) * (n * grad(v_el)) * dX

        f_el = LinearForm(VG)
        f_el += coef_f_el * v_el * ds

        with TaskManager():
            c_el = Preconditioner(a_el, "bddc")
            a_el.Assemble()
            f_el.Assemble()

            solvers.CG(mat=a_el.mat, rhs=f_el.vec, pre=c_el.mat, sol=gfu_el.vec, printrates=False)
            sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

            gfu_temp.Set(gfu_el)

        gfu_prevs.append(gfu_temp)

        if out:
            gfu_out.Set(gfu_temp)
            vtk.Do(time=t_curr)

        with TaskManager():
            l2u, h1u = errors(mesh, ds, Pmat, gfu_el, coef_u)
        append_errors(t_curr, l2u, h1u, **out_errs)

    # TIME MARCHING

    t.Set(0.0)
    t_curr = 0.0

    i = 1

    while t_curr < tfinal + dt/2:
        t.Set(t_curr + dt)
        with TaskManager():
            InterpolateToP1(phi, lset_approx)
            ci = CutInfo(mesh, lset_approx)

            ba_IF.Clear()
            ba_IF |= ci.GetElementsOfType(IF)
            update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

            VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
            u, v = VG.TnT()

        a = BilinearForm(VG)
        a += (bdf_coeff[0] * u * v +
                     dt * (1. / 2 * InnerProduct(Pmat * w, Pmat * grad(u)) * v -
                           1. / 2 * InnerProduct(Pmat * w, Pmat * grad(v)) * u +
                           (divGw - 0.5 * divGwT) * u * v +
                           nu * InnerProduct(Pmat * grad(u), Pmat * grad(v)))) * ds
        a += (dt * rho * (n * grad(u)) * (n * grad(v))) * dX

        f = LinearForm(VG)
        f += (dt * coef_f - sum([bdf_coeff[j+1] * gfu_prevs[j] for j in range(time_order)])) * v * ds

        with TaskManager():
            c = Preconditioner(a, "bddc")
            a.Assemble()
            f.Assemble()

            for j in range(time_order-1):
                gfu_prevs[-1-j].Set(gfu_prevs[-2-j])

            gfu = GridFunction(VG)

            solvers.GMRes(A=a.mat, b=f.vec, pre=c.mat, x=gfu.vec, tol=1e-15, maxsteps=5, printrates=False)

            gfu_prevs[0].Set(gfu)

            t_curr += dt
            if out:
                gfu_out.Set(gfu)
                vtk.Do(time=t_curr)

        with TaskManager():
            l2u, h1u = errors(mesh, ds, Pmat, gfu, coef_u)
        append_errors(t_curr, l2u, h1u, **out_errs)

        print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        i += 1

    return h_approx, dt, out_errs['ts'], out_errs['l2us'], out_errs['h1us']


def moving_ns_direct(order, unif_ref, bbox_sz, tfinal, time_order=1, mu=1.0, out=False, fname=None, test_name="advect"):
    bdf_coeffs = [[1, -1],
                  [3 / 2, -2, 1 / 2],
                  [11 / 6, -3, 3 / 2, -1 / 3]]

    bdf_coeff = bdf_coeffs[time_order - 1]

    c_delta = (time_order + 0.5)
    # MESH
    mesh = background_mesh(bbox_sz)

    h_approx = bbox_sz * 2 ** (1 - unif_ref)

    t = Parameter(0.0)

    dt = h_approx ** (order / time_order) / 4

    if test_name == "advect":
        vel = 0.2
        w1 = CoefficientFunction(vel)
        w2 = CoefficientFunction(0.0)
        w3 = CoefficientFunction(0.0)
        w = CoefficientFunction((w1, w2, w3))

        R = 1.0
        phi = -R + sqrt((-t*w1 + x)**2 + (-t*w2 + y)**2 + (-t*w3 + z)**2)
    else:
        vel = 0.2
        w1 = CoefficientFunction(vel)
        w2 = CoefficientFunction(0.0)
        w3 = CoefficientFunction(0.0)
        w = CoefficientFunction((w1, w2, w3))

        R = 1.0
        phi = -R + sqrt((-t * w1 + x) ** 2 + (-t * w2 + y) ** 2 + (-t * w3 + z) ** 2)

    refine_around_lset(mesh, unif_ref, phi, vel, c_delta, tfinal, dt)

    if test_name == "advect":
        wN = (-t*(w1**2 + w2**2 + w3**2) + w1*x + w2*y + w3*z)/sqrt(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)

        coef_u = CoefficientFunction((
                x * (t * w3 - z) / sqrt(
            t ** 2 * (w1 ** 2 + w2 ** 2 + w3 ** 2) - 2 * t * (w1 * x + w2 * y + w3 * z) + x ** 2 + y ** 2 + z ** 2),
                y * (-t * w3 + z) / sqrt(
                    t ** 2 * (w1 ** 2 + w2 ** 2 + w3 ** 2) - 2 * t * (w1 * x + w2 * y + w3 * z) + x ** 2 + y ** 2 + z ** 2),
                (-t * w1 * x + t * w2 * y + x ** 2 - y ** 2) / sqrt(
                    t ** 2 * (w1 ** 2 + w2 ** 2 + w3 ** 2) - 2 * t * (w1 * x + w2 * y + w3 * z) + x ** 2 + y ** 2 + z ** 2)
        ))
        coef_p = y*(-t*w1 + x) + z

        coef_f = CoefficientFunction((
            -4*mu*(t*w1 - x)*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2) + x + y + (t*w1 - x)*(t*w1*y + t*w2*x - 2*x*y)**2/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t*w1 - x)*(t**2*w1*w2 - t*(2*w1*y + w2*x + w3) + 2*x*y + z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (t*w2*x*y - x*(t**2*(w1**2 + w2**2) - 2*t*w1*x + x**2) + y**2*(-t*w1 + x))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (-t**2*w1*z*(w1**2 + w2**2 + 2*w3**2) + t*w2*w3*y*(-t*w1 + x) + t*w3*(t*w1 - 2*x)*(t*(w1**2 + w2**2 + w3**2) - w1*x) + t*z*(3*w1**2*x + 2*w1*w2*y + 3*w2**2*x + 4*w3**2*x) + w3*y**2*(-t*w1 + 2*x) + w3*z**2*(t*w1 - 2*x) - 2*x*z*(w1*x + 2*w2*y))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            4*mu*(t*w2 - y)*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2) - t*w1 + x + y + (t*w2 - y)*(t*w1*y + t*w2*x - 2*x*y)**2/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t*w2 - y)*(t**2*w1*w2 - t*(2*w1*y + w2*x + w3) + 2*x*y + z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (-t**2*y*(w1**2 + w2**2) + t*(w1*x*y - w2*x**2 + 2*w2*y**2) + y*(x - y)*(x + y))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (-t**3*w2*w3*(w1**2 + w2**2 + w3**2) + t**2*(w2*z*(w1**2 + w2**2 + 2*w3**2) + w3*(2*w1**2*y + w1*w2*x + 3*w2**2*y + 2*w3**2*y)) - t*(2*w1*w2*x*z + 3*w2**2*y*z + w2*w3*(-x**2 + 2*y**2 + z**2) + y*(3*w1**2*z + w1*w3*x + 4*w3**2*z)) + 2*y*(2*w1*x*z + w2*y*z - w3*x**2 + w3*z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            (4*mu*(t*(w1 - w2) - x + y)*(t*(w1 + w2) - x - y)*sqrt(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (t*w3 - z)*(-t**3*(w1*x + w2*y)*(w1**2 + w2**2 + w3**2) + t**2*(w1**2*(3*x**2 + 2*y**2) + 2*w1*x*(3*w2*y + w3*z) + w2**2*(2*x**2 + 3*y**2) + 2*w2*w3*y*z + w3**2*(x**2 + y**2)) - t*(w1*x*(3*x**2 + 7*y**2 + z**2) + w2*y*(7*x**2 + 3*y**2 + z**2) + 2*w3*z*(x**2 + y**2)) + x**4 + x**2*(6*y**2 + z**2) + y**2*(y**2 + z**2)) + sqrt(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)*(-t**3*(w1 - w2)*(w1 + w2)*(w1**2 + w2**2 + w3**2) + t**2*(4*w1**3*x + w1**2*(-2*w2*y + w3*z) + w1*x*(2*w2**2 + 3*w3**2) - w2*(4*w2**2*y + w2*w3*z + 3*w3**2*y)) + t*(w1**2*(-5*x**2 + 3*y**2) - 3*w1*w3*x*z + w2**2*(-3*x**2 + 5*y**2) + 3*w2*w3*y*z + 2*w3**2*(-x**2 + y**2)) + 2*w1*x**3 - 4*w1*x*y**2 + 4*w2*x**2*y - 2*w2*y**3 + 2*w3*z*(x - y)*(x + y)) + (t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)*(t**3*w1*w2*w3 + t**2*(w1*(w1 - w2*z - 2*w3*y) + w2*(w2 - w3*x)) + t*(-2*w1*x + 2*w1*y*z + w2*x*z - 2*w2*y + 2*w3*x*y) + x**2 - 2*x*y*z + y**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2
        ))

        coef_f_el = CoefficientFunction((
            (4*mu*(-t*w1 + x) + x*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            -(4*mu*(-t*w2 + y) + y*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            (4*mu*(t*(w1 - w2) - x + y)*(t*(w1 + w2) - x - y) - (t*w1*x - t*w2*y - x**2 + y**2)*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2)
        ))

        coef_g = CoefficientFunction(0.0)
    else:
        wN = (-t*(w1**2 + w2**2 + w3**2) + w1*x + w2*y + w3*z)/sqrt(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)
        coef_u = CoefficientFunction((
            -(-2*t*w3*z**3 + x*z*(-t*w1 + x) + z**4 + z**2*(t**2*(w2**2 + w3**2) - 2*t*w2*y + y**2) + (t*w1 - x)*(t*w2*y + t*w3*x - y**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2),
            (t**2*(w2*(w1*z**2 - w3*x) + y*(w1**2 + w3**2)) - t*(w1*y*(2*x + z**2) + w2*x*z*(z - 1) - w3*y*(x - 2*z)) + y*(x**2 + x*z*(z - 1) + z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2),
            (t**2*(w1**2*x + w1*w3*z**2 + w2*(w2*x - w3*y)) + t*(-w1*(2*x**2 + z**3) + w2*y*(-2*x + z) + w3*(-x*z**2 + y**2)) + x**3 + x*(y**2 + z**3) - y**2*z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)
        ))

        coef_p = CoefficientFunction(y**3*(-t*w1 + x) + z)

        coef_f = CoefficientFunction((
            mu*(2*t**4*(w1**2*(w2**2 + 2*w3**2) + w2**2*(w2**2 + w3**2)) - t**3*(w1**2*(4*w2*y + 3*w3*(6*z - 1)) + 4*w1*(w2**2*(x - 2) + 2*w3**2*x) + 8*w2**3*y + w2**2*w3*(5 - 2*z) + 4*w2*w3**2*y + w3**3*(5 - 6*z)) + t**2*(w1**2*(2*y**2 + z*(14*z - 3)) + 4*w1*w3*x*(9*z - 2) + w2**2*(x*(2*x - 8) + 12*y**2 + z*(5 - 6*z)) + 2*w2*y*(w1*(4*x - 9) + w3*(5 - 2*z)) + w3**2*(4*x**2 + 2*y**2 + z*(15 - 20*z))) + t*(2*w1*(2*x*z*(2 - 7*z) + y**2*(5 - 2*x)) - 2*w2*y*(x*(2*x - 9) + 4*y**2 + z*(5 - 6*z)) + w3*(x**2*(5 - 18*z) + y**2*(2*z - 5) + z**2*(22*z - 15))) + x**2*(2*y**2 + z*(14*z - 5)) - 10*x*y**2 + (y**2 + z**2)*(2*y**2 + z*(5 - 8*z)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + y**3 + (t*w1 - x)*(3*t*w2*y**2*(t*w1 - x) - t*w3 + y**3*(-4*t*w1 + 4*x) + z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (t*(w1**2 + w2**2 + w3**2) - w1*x - w2*y - w3*z)*(-2*t*w3*z**3 + x*z*(-t*w1 + x) + z**4 + z**2*(t**2*(w2**2 + w3**2) - 2*t*w2*y + y**2) + (t*w1 - x)*(t*w2*y + t*w3*x - y**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (-t**4*(w1*(w1*w3 + w2**2) + 2*w3*z*(w2**2 + w3**2))*(w1**2 + w2**2 + w3**2) + t**3*(w1**4*z + w1**3*(2*w2*y + 3*w3*x) + w1**2*(w2**2*(2*x + 2*z**2 + z) + w2*w3*y*(4*z + 1) + 2*w3**2*z*(3*z + 1)) + w1*(3*w2**3*y + w2**2*w3*(2*x*(z + 1) + z) + 2*w2*w3**2*y + 2*w3**3*x*(z + 1)) + (w2**2 + w3**2)*(w2**2*(x + 2*z**2) + 6*w2*w3*y*z + 8*w3**2*z**2)) - t**2*(w1**3*(3*x*z + y**2) + w1**2*(w2*y*(4*x + 5*z**2 + z) + 3*w3*x**2 + w3*z*(2*y**2 + 7*z**2 + z)) + w1*(w2**2*(x**2 + x*z*(z + 2) + 2*y**2) + w2*w3*y*(4*x*z + x + z) + w3**2*(x*z*(5*z + 3) + y**2)) + w2**3*(4*x*y + 6*y*z**2) + w2**2*w3*(2*x**2 + x*z + 6*y**2*z + 8*z**3) + w2*w3**2*y*(3*x + 14*z**2) + 2*w3**3*(x**2 + y**2*z + 6*z**3)) + t*(w1**2*(3*x**2*z + 2*x*y**2 + 3*z**2*(y**2 + z**2)) + w1*(-w2*y*(-2*x**2 - x*z*(4*z + 1) + y**2 + z**2) + w3*x*(x**2 + y**2*(2*z - 1) + 6*z**3)) + w2**2*(-x**2*z*(z - 2) + 5*x*y**2 + 2*z**2*(3*y**2 + z**2)) + w2*w3*y*(2*x**2 + 3*x*z + 2*z*(y**2 + 5*z**2)) + w3**2*(-x**2*z*(z - 4) + 2*x*y**2 + 4*z**2*(y**2 + 2*z**2))) + w1*(-x**3*z - x**2*y**2 - x*z*(y**2 + z**2)*(3*z - 1) + y**2*(y**2 + z**2)) - (w2*y + w3*z)*(-x**2*z*(z - 2) + 2*x*y**2 + 2*z**2*(y**2 + z**2)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t**4*(-w1**3*w2*y + w1**2*(-w2**2*z*(2*x + z) + w2*w3*y + w3**2*(-2*x*z + x)) + w1*w3*(w2**2*(x - 2*z**3 + z**2) - w2*w3*y + w3**2*z**2*(1 - 2*z)) - 2*w2*z*(w2**2 + w3**2)*(w2*x - w3*y)) + t**3*(w1**3*y**2 - w1**2*w3*(x*z*(2 - 5*z) + y**2) + w1*w3**2*(x**2*(4*z - 3) + y**2 + z**3*(7*z - 3)) + w2**3*y*z*(8*x - 3*z) - w2**2*(w1*y**2 + w1*z*(-4*x**2 - 2*x*z + x - 3*z**3 + z**2) + w3*x*(x - 2*z**2*(z + 1)) + 6*w3*y**2*z) + w2*y*(w1**2*(x*(4*z + 3) + z*(3*z - 1)) + 2*w1*w3*(-3*x + z**2*(2*z - 1) + z) + w3**2*(4*x*z + x - 7*z**2)) + 2*w3**3*z*(x*z*(z - 1) - y**2)) - t**2*(w1**2*(x*y**2*(2*z + 3) + x*z**2*(3*z - 1) + y**2*z*(2*z - 1)) + w1*w2*y*(x**2*(8*z + 3) + 6*x*z*(z - 1) - 2*y**2 + 6*z**4 - 2*z**3 + z**2) + w2**2*(2*x**3*z + x**2*z*(z - 1) + x*y**2*(12*z - 1) + 3*x*z**4 - 9*y**2*z**2) + w3**2*(2*x**3*(z - 1) + x*y**2*(2*z + 1) + x*z**3*(7*z - 6) - 7*y**2*z**2) + w3*(w1*(2*x**2*z*(5*z - 3) - 5*x*y**2 + y**2*z*(z*(2*z - 1) + 2) + z**4*(8*z - 3)) + w2*y*(-5*x**2 + 2*x*z*(2*z*(z + 1) + 1) - 6*y**2*z - 8*z**3))) + t*(w1*(x**2*(y**2*(4*z + 3) + 3*z**2*(2*z - 1)) + x*y**2*z*(4*z - 5) - y**4 + y**2*z**2*(z*(3*z - 1) + 1) + z**5*(3*z - 1)) + w2*y*(x**3*(4*z + 1) + x**2*z*(3*z - 5) + x*(y**2*(8*z - 2) + 6*z**4 + z**2) - 9*y**2*z**2 - 3*z**4) + w3*(x**3*z*(5*z - 4) - 4*x**2*y**2 + 2*x*z*(y**2*(z**2 + z + 1) + z**3*(4*z - 3)) - 2*y**2*z*(y**2 + 4*z**2))) - x**3*(y**2*(2*z + 1) + z**2*(3*z - 2)) - 2*x**2*y**2*z*(z - 2) + x*(y**4*(1 - 2*z) - y**2*(3*z**4 + z**2) + z**5*(2 - 3*z)) + 3*y**2*z**2*(y**2 + z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            -2*mu*(t**4*w1*w2*(w1**2 + w2**2 - w3**2) + t**3*(-w1**3*y + w1**2*w2*(4 - 3*x) + w1*(-3*w2**2*y + 2*w2*w3*(5*z - 2) + w3**2*y) - w2**3*x + w2*w3**2*(x + 4)) + t**2*(w1**2*y*(3*x - 5) + w1*w2*(x*(3*x - 8) + 3*y**2 + z*(4 - 10*z)) + 2*w1*w3*y*(2 - 5*z) + 3*w2**2*x*y + 5*w2*w3*x - 2*w2*w3*z*(5*x + 4) - w3**2*y*(x + 5)) + 2*t*z**2*(5*w1*y + w2*(5*x + 2)) + t*z*(-4*w1*y - 5*w2*x + 10*w3*y*(x + 1)) - t*(w1*y*(x*(3*x - 10) + y**2) + w2*x*(x*(x - 4) + 3*y**2) + 5*w3*x*y) + y*(5*x*z + x*(x*(x - 5) + y**2) - z**2*(10*x + 5)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + y**2*(-3*t*w1 + 3*x) + (t*w2 - y)*(3*t*w2*y**2*(t*w1 - x) - t*w3 + y**3*(-4*t*w1 + 4*x) + z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) - (t**2*(w2*(w1*z**2 - w3*x) + y*(w1**2 + w3**2)) - t*(w1*y*(2*x + z**2) + w2*x*z*(z - 1) - w3*y*(x - 2*z)) + y*(x**2 + x*z*(z - 1) + z**2))*(t*(w1**2 + w2**2 + w3**2) - w1*x - w2*y - w3*z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t**4*(w1**4*y + w1**3*w2*z*(2*x + z) + w1**2*w3*(-w2*x + 2*w2*z**3 + 2*w3*y) + w1*w2*(2*w2**2*x*z + w2*w3*y*(1 - 2*z) + w3**2*(x + z**2)) + w2**3*w3*z**2 + w2*w3**3*(-x + z**2) + w3**4*y) + t**3*(-2*w1**3*y*(x*(z + 2) + z**2) + w1**2*(w2*(y**2 + z*(-6*x**2 - 3*x*z + x - 3*z**3)) + 2*w3*y*(x - z*(z**2 + 2))) + w1*(w2**2*y*z*(-6*x + 3*z - 1) + 2*w2*w3*(x**2 + x*z*(-2*z**2 + z - 1) + y**2*(2*z - 1) - z**3) - w3**2*y*(5*x + 2*z**2)) - w2**3*(2*x**2*z + z**3) + w2**2*w3*y*(2*x*(z - 1) - 3*z**2) + w2*w3**2*(-2*x**2 - x*z*(z - 3) + y**2 - 3*z**3) + w3**3*y*(2*x - z*(z + 4))) + t**2*(w2**2*y*z*(6*x**2 + x*(2 - 3*z) + 3*z**2) + w2*(-w3*x**3 - x*y**2*(2*w1 - 4*w3) + 2*x*z**3*(-w1 + w3*x + w3) + z**4*(6*w1*x + w1 + 3*w3) + z**2*(x*(3*w1*x + w1 - w3*(2*x + 3)) + y**2*(-6*w1 + 3*w3)) + z*(2*x**2*(w1*(3*x - 1) + 2*w3) + 2*y**2*(3*w1*x + w1 - w3*(2*x + 1)))) + y*(w1**2*(6*x**2*(z + 1) + 2*x*z*(3*z - 1) - y**2 + 3*z**4 + 2*z**2) + w1*w3*(-4*x**2 + 2*x*z*(z*(2*z - 1) + 5) - 2*y**2*z + y**2 + 4*z**3) + w3**2*(4*x**2 + 2*x*z*(z - 3) - y**2 + 3*z**2*(z + 2)))) + t*(-w2*z**5 + x*y*(w2*x*y + x**2*(-4*w1 + 2*w3) + y**2*(2*w1 - 2*w3)) - z**4*(w1*y*(6*x + 2) + w2*x*(3*x + 1) + 3*w3*y) + z**3*(2*w1*x*y + w2*(2*x**2 + x - 3*y**2) - 2*w3*y*(x*(x + 2) + 2)) + z**2*(-w2*x**2*(x + 2) + w2*y**2*(6*x + 1) + x*y*(-w1*(6*x + 5) + 2*w3*(x + 3)) + y**3*(3*w1 - w3)) - z*(w2*x**3*(2*x - 1) + 2*w2*x*y**2*(3*x + 2) + 2*x**2*y*(w1*(3*x - 2) + 4*w3) + y**3*(2*w1*x + w1 - 2*w3*(x + 1)))) + y*(x**2*(x - y)*(x + y) + 2*x*z*(x**2*(x - 1) + y**2*(x + 1)) + z**5 + z**4*(x*(3*x + 2) + 1) + z**3*(-2*x*(x + 1) + y**2) + z**2*(2*x**2*(x + 2) - y**2*(3*x + 1))))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t**4*w2*(w1**2 + w2**2 + w3**2)*(w1**2 + w1*w3*(2*z - 1) + w3**2) + t**3*(-w1**4*y + w1**3*(w2*(-3*x - 2*z**2 + z) + w3*y*(1 - 2*z)) - w1**2*(-2*w2*w3*x + w2*w3*z*(4*x + 3) + y*(2*w2**2 + 2*w3**2)) + w1*(w2**3*(-2*x - 2*z**2 + z) + 2*w2**2*w3*y*(1 - 2*z) + w2*w3**2*(-3*x - 4*z**2 + 2*z) + w3**3*y*(1 - 2*z)) - w3*(w2**3*(-x + z*(2*x + 2)) + 2*w2**2*w3*y + w2*w3**2*(2*x*z - x + 3*z) + w3**3*y)) + t**2*(w1**3*y*(3*x + z*(3*z - 1)) + w1**2*(w2*(3*x**2 + x*z*(3*z - 2) + z**2) + w3*y*(x*(4*z - 3) + 3*z)) + w1*(w2**2*y*(5*x + 2*z*(2*z - 1)) + w2*w3*(y**2*(2*z - 1) + z*(2*x*(x + 2) + z*(z - 1))) + w3**2*y*(3*x + z*(5*z - 2))) + w2*z**2*(w2**2*(2*x + 1) + w3**2*(4*x + 3)) + x*(w2**2 + w3**2)*(w2*x - 2*w3*y) + z*(w2**2*w3*y*(4*x + 5) - w2*x*(w2**2 + w3**2) + w3**3*y*(2*x + 3))) + t*(w1**2*y*(-3*x**2 + 3*x*z + y**2 - z**2*(6*x + 1)) + w1*(-w2*(x**3 + x*(2*y**2 + z**2) - z*(-2*y**2*z + y**2 + z**3)) + w3*y*(2*x**2 - 2*x*z*(x + 2) - 3*z**3 + z**2)) - w2**2*y*(3*x**2 - 2*x*z + z**2*(4*x + 3)) - w2*w3*(x**3 + x**2*z + x*y**2*(2*z - 1) + x*z**2*(z + 1) + 2*y**2*z + z**3) + w3**2*y*(-x**2 + x*z*(4 - 5*z) + y**2 - 3*z**2)) + w1*x*y*(x**2 + x*z*(3*z - 2) - y**2 + z**2) + 2*w2*x**2*y**2 - w2*x*z**4 + z**3*(w2*x + w3*y*(3*x + 1)) + z**2*(-w2*x**3 + 2*w2*y**2*(x + 1) - 2*w3*x*y) + z*(x - y)*(x + y)*(w2*x + w3*y))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            mu*(-4*t**4*w1*w3*(w1**2 + w2**2) + t**3*(w1**3*(14*z - 3) + 12*w1**2*w3*x + w1*(w2**2*(14*z - 3) + 8*w2*w3*y + w3**2*(5 - 6*z)) + 4*w2**2*w3*(x + 2)) - t**2*(w1**2*x*(42*z - 11) + 2*w1*(w2*y*(14*z - 3) + w3*(6*x**2 + 2*y**2 + z*(5 - 7*z))) + w2**2*(14*x*z - 5*x + 8*z) + 2*w2*w3*y*(4*x + 9) + w3**2*x*(5 - 6*z)) + t*(w1*x**2*(42*z - 13) + w1*y**2*(14*z - 3) + w1*z**2*(5 - 8*z) + 2*w2*y*(x*(14*z - 5) + 9*z) + 2*w3*(2*x**3 + 2*x*y**2 + x*z*(5 - 7*z) + 5*y**2)) + 8*x*z**3 - 5*x*z**2 + 5*x*(x**2 + y**2) - z*(14*x**3 + 2*y**2*(7*x + 5)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 - (t*(w1**2 + w2**2 + w3**2) - w1*x - w2*y - w3*z)*(t**2*(w1**2*x + w1*w3*z**2 + w2*(w2*x - w3*y)) + t*(-w1*(2*x**2 + z**3) + w2*y*(-2*x + z) + w3*(-x*z**2 + y**2)) + x**3 + x*(y**2 + z**3) - y**2*z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (3*t**3*w1*w2*w3*y**2 + t**2*(w1**2 - w1*y**2*(3*w2*z + 4*w3*y) + w2*(w2 - 3*w3*x*y**2)) + t*(-2*w1*(x - 2*y**3*z) + w2*y*(3*x*y*z - 2) + 4*w3*x*y**3) + x**2 - 4*x*y**3*z + y**2)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (t**4*(-w1**3*(w2*y + w3*(-2*x*z + x)) - w1**2*(w2**2*z**2 + w2*w3*y + w3**2*z**2*(1 - 2*z)) - w1*w2*(w2**2*y + w2*w3*(-2*x*z + x + z**2) + 2*w3**2*y*z) + w2*(-w2*z**2*(w2**2 + w3**2) + w3**2*(w2*x - w3*y))) + t**3*(w1**3*(x*z*(1 - 3*z) + y**2) + w1**2*(w2*y*(4*x + 2*z**2 + z) + w3*(x**2*(4 - 6*z) + y**2 + z**3*(2 - 5*z))) + w1*(w2**2*(3*y**2 + z*(-x*z + x + z**2)) + 4*w2*w3*y*(-x*z + x + 2*z**2) + w3**2*z*(x*z*(3 - 4*z) + 2*y**2)) + 2*w2**3*y*(x + 2*z**2) + w2**2*w3*(-2*x**2*(z - 1) + x*z*(z - 2) - y**2 + 2*z**3) + w2*w3**2*y*(x*(2*z - 3) + z*(2*z + 3)) + w3**3*y**2) + t**2*(w1**2*(x**2*z*(9*z - 4) - 4*x*y**2 - y**2*z*(z + 1) + z**4*(3*z - 1)) + w1*(-w2*y*(5*x**2 - 2*x*z*(z - 2) + 3*y**2 + 6*z**3) + w3*(x**3*(6*z - 5) + x*y**2*(2*z - 3) + 2*x*z**3*(5*z - 3) - 7*y**2*z**2)) + w2**2*(2*x**2*z*(z - 1) + x*(-6*y**2 - z**3 + z**2) + y**2*z*(1 - 6*z) - z**4) + w2*w3*y*(x**2*(4*z - 5) + 2*x*z*(3 - 4*z) + 2*y**2 - z**2*(4*z + 3)) + w3**2*(2*x**2*z**2*(z - 1) - 2*x*y**2*(z - 1) - y**2*z*(z + 3))) + t*(w1*(x**3*z*(5 - 9*z) + 5*x**2*y**2 - x*y**2*z*(z - 3) + 3*x*z**4*(1 - 2*z) + y**4 + 5*y**2*z**3) + w2*y*(2*x**3 + x**2*z*(5 - 4*z) + 6*x*y**2 + 3*x*z**2*(2*z - 1) + y**2*z*(4*z - 2) + z**3*(2*z + 1)) + w3*(-2*x**4*(z - 1) + x**2*(y**2*(3 - 2*z) + z**3*(4 - 5*z)) + x*y**2*z*(7*z - 4) - y**4 + y**2*z**2*(2*z + 3))) + x**4*z*(3*z - 2) - 2*x**3*y**2 + x**2*z*(y**2*(2*z - 3) + z**3*(3*z - 2)) + x*y**2*(-2*y**2 + z**2*(2 - 5*z)) - y**2*z*(y**2*(z - 1) + z**2*(z + 1)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t**4*(w1**2 + w2**2 + w3**2)*(w1**3 + w1*(w2**2 + 2*w3**2*z) - w2**2*w3) + t**3*(-4*w1**4*x - w1**3*(3*w2*y + w3*z*(4*z + 1)) + w1**2*(w2**2*(-5*x + z) + 2*w2*w3*y - w3**2*x*(4*z + 3)) - w1*(3*w2**3*y + w2**2*w3*(-x + 4*z**2 + z) + 2*w2*w3**2*y*(z + 1) + 6*w3**3*z**2) + w2**4*(-x + z) + 3*w2**3*w3*y - w2**2*w3**2*(x + z*(2*x - 2)) + 2*w2*w3**3*y - 2*w3**4*x*z) + t**2*(w1**3*(6*x**2 + y**2 + 3*z**3) + w1**2*(w2*y*(7*x - 3*z) + w3*x*z*(7*z + 2) - w3*y**2) + w1*(w2**2*(4*x**2 - x*z + 3*y**2 + 3*z**3) + w2*w3*y*(-x + z*(3*z + 2)) + w3**2*(2*x**2*(z + 2) + y**2 + 6*z**3)) + w2**3*y*(3*x - 4*z) - w2**2*w3*(-4*x*z**2 + 2*y**2 + z**2) + w2*w3**2*y*(3*x + z*(2*x - 4)) - w3**3*(-6*x*z**2 + y**2)) + t*(w1**2*(-4*x**3 + x*(-2*y**2 - 6*z**3 + z**2) + 2*y**2*z) - w1*(w2*y*(5*x**2 - 3*x*z + y**2 + 3*z**3) + w3*z*(x**2*(2*z + 3) - y**2*z + y**2 + 2*z**3)) - w2**2*(x**3 + 3*x*y**2 + x*z**2*(3*z - 1) - 5*y**2*z) - w2*w3*y*(x**2 + x*z*(3*z + 2) + y**2 - 2*z**2) - 2*w3**2*(x**3 + x*(y**2 + 3*z**3) - y**2*z)) + 2*w3*x*z**4 + 3*x*z**3*(w1*x + w2*y) - z**2*(w2*x*y + w3*y**2*(x + 1) + x**2*(w1 + w3*x)) + z*(2*w3*x*(x**2 + y**2) - 2*y**2*(w1*x + w2*y)) + (x**2 + y**2)*(w1*x**2 + y*(w2*x + w3*y)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2
        ))

        coef_f_el = CoefficientFunction((
            (mu*(2*t**4*(w1**2*(w2**2 + 2*w3**2) + w2**2*(w2**2 + w3**2)) - t**3*(w1**2*(4*w2*y + 3*w3*(6*z - 1)) + 4*w1*(w2**2*(x - 2) + 2*w3**2*x) + 8*w2**3*y + w2**2*w3*(5 - 2*z) + 4*w2*w3**2*y + w3**3*(5 - 6*z)) + t**2*(w1**2*(2*y**2 + z*(14*z - 3)) + 4*w1*w3*x*(9*z - 2) + w2**2*(x*(2*x - 8) + 12*y**2 + z*(5 - 6*z)) + 2*w2*y*(w1*(4*x - 9) + w3*(5 - 2*z)) + w3**2*(4*x**2 + 2*y**2 + z*(15 - 20*z))) + t*(-4*w1*x*z*(7*z - 2) - 2*w1*y**2*(2*x - 5) - 2*w2*y*(x*(2*x - 9) + 4*y**2 + z*(5 - 6*z)) + w3*(x**2*(5 - 18*z) + y**2*(2*z - 5) + z**2*(22*z - 15))) + x**2*(2*y**2 + z*(14*z - 5)) - 10*x*y**2 + 2*y**4 - 6*y**2*z**2 + 5*y**2*z - 8*z**4 + 5*z**3) - (t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)*(-2*t*w3*z**3 + x*z*(-t*w1 + x) + z**4 + z**2*(t**2*(w2**2 + w3**2) - 2*t*w2*y + y**2) + (t*w1 - x)*(t*w2*y + t*w3*x - y**2)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            (-2*mu*(t**4*w1*w2*(w1**2 + w2**2 - w3**2) + t**3*(-w1**3*y + w1**2*w2*(4 - 3*x) + w1*(-3*w2**2*y + 2*w2*w3*(5*z - 2) + w3**2*y) - w2**3*x + w2*w3**2*(x + 4)) + t**2*(w1**2*y*(3*x - 5) + w1*w2*(x*(3*x - 8) + 3*y**2 + z*(4 - 10*z)) + 2*w1*w3*y*(2 - 5*z) + 3*w2**2*x*y + w2*w3*(5*x - z*(10*x + 8)) - w3**2*y*(x + 5)) - t*(w1*y*(x*(3*x - 10) + y**2 + z*(4 - 10*z)) + 5*w2*x*z + w2*x*(x*(x - 4) + 3*y**2) - 2*w2*z**2*(5*x + 2) + 5*w3*y*(x - z*(2*x + 2))) + y*(5*x*z + x*(x*(x - 5) + y**2) - z**2*(10*x + 5))) + (t**2*(w2*(w1*z**2 - w3*x) + y*(w1**2 + w3**2)) - t*(w1*y*(2*x + z**2) + w2*x*z*(z - 1) - w3*y*(x - 2*z)) + y*(x**2 + x*z*(z - 1) + z**2))*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            (-mu*(4*t**4*w1*w3*(w1**2 + w2**2) - t**3*(w1**3*(14*z - 3) + 12*w1**2*w3*x + w1*(w2**2*(14*z - 3) + 8*w2*w3*y + w3**2*(5 - 6*z)) + 4*w2**2*w3*(x + 2)) + t**2*(w1**2*x*(42*z - 11) + 2*w1*(w2*y*(14*z - 3) + w3*(6*x**2 + 2*y**2 + z*(5 - 7*z))) + w2**2*(14*x*z - 5*x + 8*z) + 2*w2*w3*y*(4*x + 9) + w3**2*x*(5 - 6*z)) - t*(w1*x**2*(42*z - 13) + w1*y**2*(14*z - 3) + w1*z**2*(5 - 8*z) + 2*w2*y*(x*(14*z - 5) + 9*z) + 2*w3*(2*x**3 + 2*x*y**2 + x*z*(5 - 7*z) + 5*y**2)) + x**3*(14*z - 5) + x*y**2*(14*z - 5) + x*z**2*(5 - 8*z) + 10*y**2*z) + (t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)*(t**2*(w1**2*x + w1*w3*z**2 + w2*(w2*x - w3*y)) + t*(-w1*(2*x**2 + z**3) + w2*y*(-2*x + z) + w3*(-x*z**2 + y**2)) + x**3 + x*(y**2 + z**3) - y**2*z))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2
        ))

        coef_g = CoefficientFunction((t**2*(w1**2 + w1*w3*(2*z - 1) + w3**2) + t*(w1*(-2*x - 4*z**2 + z) + 2*w2*y + 3*w3*x - 2*w3*z*(x + 1)) + x**2 + x*z*(4*z - 3) - 2*y**2 + z**2)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))

    # Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(phi)
    lset_approx = GridFunction(H1(mesh, order=1, dirichlet=[]))
    InterpolateToP1(phi, lset_approx)
    ci = CutInfo(mesh, lset_approx)

    ba_IF = ci.GetElementsOfType(IF)
    ba_IF_band = BitArray(mesh.ne)
    update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

    # FESpace: Taylor-Hood element
    V = VectorH1(mesh, order=order, dirichlet=[])
    Q = H1(mesh, order=order - 1, dirichlet=[])

    # define projection matrix
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ba_IF, deformation=deformation)
    dX = dx(definedonelements=ba_IF_band, deformation=deformation)
    dX2 = dx(definedonelements=ba_IF, deformation=deformation)

    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)
    h = specialcf.mesh_size
    tau = 1.0 / (h * h)
    rho_u = 1.0 / h
    rho_p = 1.0 * h

    gfu_prevs = [GridFunction(V) for i in range(time_order)]

    out_errs = {'ts': [], 'l2us': [], 'h1us': [], 'l2ps': [], 'h1ps': []}

    if out:
        gfu_out = GridFunction(V)
        gfp_out = GridFunction(Q)
        if fname:
            filename = f"./vtk_out/diffusion/moving-ns-{fname}"
        else:
            filename = "./vtk_out/diffusion/moving-ns"
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, phi, gfu_out, coef_u, gfp_out, coef_p],
                        names=["P1-levelset", "phi", "u", "uSol", "p", "pSol"],
                        filename=filename,
                        subdivision=0)

    for j in range(time_order):
        # fix levelset
        t.Set(-j * dt)
        t_curr = -j * dt

        deformation = lsetmeshadap.CalcDeformation(phi)

        # solve elliptic problem on a fixed surface to get u with normal extension
        InterpolateToP1(phi, lset_approx)
        ci = CutInfo(mesh, lset_approx)

        ba_IF.Clear()
        ba_IF |= ci.GetElementsOfType(IF)
        update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

        VG = Compress(V, GetDofsOfElements(V, ba_IF_band))

        # helper grid functions
        n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=phi, vel_space=V)

        gfu_el = GridFunction(VG)

        u_el, v_el = VG.TnT()

        a_el = BilinearForm(VG, symmetric=True)
        a_el += InnerProduct(Pmat * u_el, Pmat * v_el) * ds
        a_el += 2 * mu * (InnerProduct(Pmat * Sym(grad(u_el)) * Pmat - (u_el * n) * Hmat,
                                       Pmat * Sym(grad(v_el)) * Pmat - (v_el * n) * Hmat)) * ds
        a_el += (tau * ((u_el * n_k) * (v_el * n_k))) * ds
        a_el += (rho_u * InnerProduct(grad(u_el) * n, grad(v_el) * n)) * dX

        f_el = LinearForm(VG)
        f_el += InnerProduct(coef_f_el, Pmat * v_el) * ds

        with TaskManager():
            pre_a_el = Preconditioner(a_el, "bddc")

            a_el.Assemble()
            f_el.Assemble()

            solvers.CG(mat=a_el.mat, rhs=f_el.vec, pre=pre_a_el.mat, sol=gfu_el.vec, printrates=False)
            sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

            gfu_prevs[j].Set(gfu_el)

        with TaskManager():
            l2u, h1u = errors_u(mesh, ds, Pmat, gfu_prevs[j], coef_u)
        append_errors(t_curr, l2u, h1u, 0, 0, **out_errs)

    # TIME MARCHING
    t.Set(0.0)
    t_curr = 0.0

    if out:
        gfu_out.Set(gfu_prevs[0])
        gfp_out.Set(coef_p)
        vtk.Do(time=t_curr)

    i = 1

    l2err_old = 0.0

    while t_curr < tfinal - dt:
        t.Set(t_curr + dt)
        with TaskManager():
            deformation = lsetmeshadap.CalcDeformation(phi)

            InterpolateToP1(phi, lset_approx)
            ci = CutInfo(mesh, lset_approx)

            ba_IF.Clear()
            ba_IF |= ci.GetElementsOfType(IF)
            update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

            VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
            QG = Compress(Q, GetDofsOfElements(Q, ba_IF))
            XG = FESpace([VG, QG])
            u, p = XG.TrialFunction()
            v, q = XG.TestFunction()

            # helper grid functions
            n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=phi, vel_space=V)

        gfu_approx = GridFunction(VG)
        if time_order == 1:
            gfu_approx.Set(Pmat * gfu_prevs[0])
        elif time_order == 2:
            gfu_approx.Set(2 * Pmat * gfu_prevs[0] - Pmat * gfu_prevs[1])
        else:
            gfu_approx.Set(3 * Pmat * gfu_prevs[0] - 3 * Pmat * gfu_prevs[1] + Pmat * gfu_prevs[2])

        a = BilinearForm(XG, symmetric=False)
        a += bdf_coeff[0] / dt * InnerProduct(u, Pmat * v) * ds
        a += wN * InnerProduct(Hmat * u, Pmat * v) * ds
        a += 0.5 * InnerProduct((Pmat * grad(u) * Pmat - (u * n) * Hmat) * gfu_approx, v) * ds
        a += (-0.5) * InnerProduct((Pmat * grad(v) * Pmat - (v * n) * Hmat) * gfu_approx, u) * ds
        a += (-0.5) * InnerProduct(coef_g * u, Pmat * v) * ds
        a += 2.0 * mu * (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                                      Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        a += tau * InnerProduct(n_k, u) * InnerProduct(n_k, v) * ds
        a += rho_u * InnerProduct(grad(u) * n, grad(v) * n) * dX

        a += InnerProduct(u, Pmat * grad(q)) * ds
        a += InnerProduct(v, Pmat * grad(p)) * ds

        a += (-1.0) * rho_p * (grad(p) * n) * (grad(q) * n) * dX2

        f = LinearForm(XG)
        f += InnerProduct(coef_f, Pmat * v) * ds
        for j in range(time_order):
            f += (-1.0) * bdf_coeff[j + 1] / dt * InnerProduct(gfu_prevs[j], Pmat * v) * ds
        f += (-1.0) * coef_g * q * ds

        with TaskManager():
            a.Assemble()
            f.Assemble()

            gf = GridFunction(XG)

        gf.vec.data = a.mat.Inverse(freedofs=XG.FreeDofs(), inverse="umfpack") * f.vec

        gfu = gf.components[0]

        l2err = sqrt(Integrate(InnerProduct(Pmat * (gfu - gfu_prevs[0]), Pmat * (gfu - gfu_prevs[0])) * ds, mesh))

        if i > 1 and l2err > 5 * l2err_old:
            continue

        for j in range(time_order-1):
            gfu_prevs[-1 - j].vec.data = gfu_prevs[-2 - j].vec

        gfp = gf.components[1]
        # making numerical pressure mean zero
        renormalize(QG, mesh, ds, gfp)

        if out:
            gfu_out.Set(gfu)
            gfp_out.Set(gfp)
            vtk.Do(time=t_curr+dt)

        l2u, h1u, l2p, h1p = errors(mesh, ds, Pmat, gfu, gfp, coef_u, coef_p)

        gfu_prevs[0].Set(gfu)
        append_errors(t_curr, l2u, h1u, l2p, h1p, **out_errs)

        print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        l2err_old = l2err
        t_curr += dt
        i += 1

    return h_approx, dt, out_errs['ts'], out_errs['l2us'], out_errs['h1us'], out_errs['l2ps'], out_errs['h1ps']


def moving_ns(order, unif_ref, bbox_sz, tfinal, time_order=1, mu=1.0, out=False, fname=None, test_name="advect"):
    bdf_coeffs = [[1, -1],
                  [3 / 2, -2, 1 / 2],
                  [11 / 6, -3, 3 / 2, -1 / 3]]

    bdf_coeff = bdf_coeffs[time_order - 1]

    c_delta = (time_order + 0.1)
    # MESH
    mesh = background_mesh(bbox_sz)

    h_approx = bbox_sz * 2 ** (1 - unif_ref)

    t = Parameter(0.0)

    dt = h_approx ** (order / time_order) / 4

    if test_name == "advect":
        vel = 0.2
        w1 = CoefficientFunction(vel)
        w2 = CoefficientFunction(0.0)
        w3 = CoefficientFunction(0.0)
        w = CoefficientFunction((w1, w2, w3))

        R = 1.0
        phi = -R + sqrt((-t*w1 + x)**2 + (-t*w2 + y)**2 + (-t*w3 + z)**2)
        refine_around_lset(mesh, unif_ref, phi, vel, c_delta, tfinal, dt)
    elif test_name == 'advect-nonsol':
        vel = 0.2
        w1 = CoefficientFunction(vel)
        w2 = CoefficientFunction(0.0)
        w3 = CoefficientFunction(0.0)
        w = CoefficientFunction((w1, w2, w3))

        R = 1.0
        phi = -R + sqrt((-t * w1 + x) ** 2 + (-t * w2 + y) ** 2 + (-t * w3 + z) ** 2)
        refine_around_lset(mesh, unif_ref, phi, vel, c_delta, tfinal, dt)
    else:
        vel = 0.5
        R = 1.0
        phi = -R + (2*t*(x**2*y**2 + x**2*z**2 + y**2*z**2) + x**4 + y**4 + z**4)**(1/4)
        refine_around_lset(mesh, unif_ref, phi, vel, c_delta, tfinal, dt, band_type="inner")

    if test_name == "advect":
        wN = (-t*(w1**2 + w2**2 + w3**2) + w1*x + w2*y + w3*z)/sqrt(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)

        coef_u = CoefficientFunction((
                x * (t * w3 - z) / sqrt(
            t ** 2 * (w1 ** 2 + w2 ** 2 + w3 ** 2) - 2 * t * (w1 * x + w2 * y + w3 * z) + x ** 2 + y ** 2 + z ** 2),
                y * (-t * w3 + z) / sqrt(
                    t ** 2 * (w1 ** 2 + w2 ** 2 + w3 ** 2) - 2 * t * (w1 * x + w2 * y + w3 * z) + x ** 2 + y ** 2 + z ** 2),
                (-t * w1 * x + t * w2 * y + x ** 2 - y ** 2) / sqrt(
                    t ** 2 * (w1 ** 2 + w2 ** 2 + w3 ** 2) - 2 * t * (w1 * x + w2 * y + w3 * z) + x ** 2 + y ** 2 + z ** 2)
        ))
        coef_p = y*(-t*w1 + x) + z

        coef_f = CoefficientFunction((
            -4*mu*(t*w1 - x)*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2) + x + y + (t*w1 - x)*(t*w1*y + t*w2*x - 2*x*y)**2/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t*w1 - x)*(t**2*w1*w2 - t*(2*w1*y + w2*x + w3) + 2*x*y + z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (t*w2*x*y - x*(t**2*(w1**2 + w2**2) - 2*t*w1*x + x**2) + y**2*(-t*w1 + x))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (-t**2*w1*z*(w1**2 + w2**2 + 2*w3**2) + t*w2*w3*y*(-t*w1 + x) + t*w3*(t*w1 - 2*x)*(t*(w1**2 + w2**2 + w3**2) - w1*x) + t*z*(3*w1**2*x + 2*w1*w2*y + 3*w2**2*x + 4*w3**2*x) + w3*y**2*(-t*w1 + 2*x) + w3*z**2*(t*w1 - 2*x) - 2*x*z*(w1*x + 2*w2*y))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            4*mu*(t*w2 - y)*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2) - t*w1 + x + y + (t*w2 - y)*(t*w1*y + t*w2*x - 2*x*y)**2/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t*w2 - y)*(t**2*w1*w2 - t*(2*w1*y + w2*x + w3) + 2*x*y + z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (-t**2*y*(w1**2 + w2**2) + t*(w1*x*y - w2*x**2 + 2*w2*y**2) + y*(x - y)*(x + y))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (-t**3*w2*w3*(w1**2 + w2**2 + w3**2) + t**2*(w2*z*(w1**2 + w2**2 + 2*w3**2) + w3*(2*w1**2*y + w1*w2*x + 3*w2**2*y + 2*w3**2*y)) - t*(2*w1*w2*x*z + 3*w2**2*y*z + w2*w3*(-x**2 + 2*y**2 + z**2) + y*(3*w1**2*z + w1*w3*x + 4*w3**2*z)) + 2*y*(2*w1*x*z + w2*y*z - w3*x**2 + w3*z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            (4*mu*(t*(w1 - w2) - x + y)*(t*(w1 + w2) - x - y)*sqrt(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (t*w3 - z)*(-t**3*(w1*x + w2*y)*(w1**2 + w2**2 + w3**2) + t**2*(w1**2*(3*x**2 + 2*y**2) + 2*w1*x*(3*w2*y + w3*z) + w2**2*(2*x**2 + 3*y**2) + 2*w2*w3*y*z + w3**2*(x**2 + y**2)) - t*(w1*x*(3*x**2 + 7*y**2 + z**2) + w2*y*(7*x**2 + 3*y**2 + z**2) + 2*w3*z*(x**2 + y**2)) + x**4 + x**2*(6*y**2 + z**2) + y**2*(y**2 + z**2)) + sqrt(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)*(-t**3*(w1 - w2)*(w1 + w2)*(w1**2 + w2**2 + w3**2) + t**2*(4*w1**3*x + w1**2*(-2*w2*y + w3*z) + w1*x*(2*w2**2 + 3*w3**2) - w2*(4*w2**2*y + w2*w3*z + 3*w3**2*y)) + t*(w1**2*(-5*x**2 + 3*y**2) - 3*w1*w3*x*z + w2**2*(-3*x**2 + 5*y**2) + 3*w2*w3*y*z + 2*w3**2*(-x**2 + y**2)) + 2*w1*x**3 - 4*w1*x*y**2 + 4*w2*x**2*y - 2*w2*y**3 + 2*w3*z*(x - y)*(x + y)) + (t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)*(t**3*w1*w2*w3 + t**2*(w1*(w1 - w2*z - 2*w3*y) + w2*(w2 - w3*x)) + t*(-2*w1*x + 2*w1*y*z + w2*x*z - 2*w2*y + 2*w3*x*y) + x**2 - 2*x*y*z + y**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2
        ))

        coef_f_el = CoefficientFunction((
            (4*mu*(-t*w1 + x) + x*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            -(4*mu*(-t*w2 + y) + y*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            (4*mu*(t*(w1 - w2) - x + y)*(t*(w1 + w2) - x - y) - (t*w1*x - t*w2*y - x**2 + y**2)*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2)
        ))

        coef_g = CoefficientFunction(0.0)
    elif test_name == 'advect-nonsol':
        wN = (-t*(w1**2 + w2**2 + w3**2) + w1*x + w2*y + w3*z)/sqrt(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)
        coef_u = CoefficientFunction((
            -(-2*t*w3*z**3 + x*z*(-t*w1 + x) + z**4 + z**2*(t**2*(w2**2 + w3**2) - 2*t*w2*y + y**2) + (t*w1 - x)*(t*w2*y + t*w3*x - y**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2),
            (t**2*(w2*(w1*z**2 - w3*x) + y*(w1**2 + w3**2)) - t*(w1*y*(2*x + z**2) + w2*x*z*(z - 1) - w3*y*(x - 2*z)) + y*(x**2 + x*z*(z - 1) + z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2),
            (t**2*(w1**2*x + w1*w3*z**2 + w2*(w2*x - w3*y)) + t*(-w1*(2*x**2 + z**3) + w2*y*(-2*x + z) + w3*(-x*z**2 + y**2)) + x**3 + x*(y**2 + z**3) - y**2*z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)
        ))

        coef_p = CoefficientFunction(y**3*(-t*w1 + x) + z)

        coef_f = CoefficientFunction((
            mu*(2*t**4*(w1**2*(w2**2 + 2*w3**2) + w2**2*(w2**2 + w3**2)) - t**3*(w1**2*(4*w2*y + 3*w3*(6*z - 1)) + 4*w1*(w2**2*(x - 2) + 2*w3**2*x) + 8*w2**3*y + w2**2*w3*(5 - 2*z) + 4*w2*w3**2*y + w3**3*(5 - 6*z)) + t**2*(w1**2*(2*y**2 + z*(14*z - 3)) + 4*w1*w3*x*(9*z - 2) + w2**2*(x*(2*x - 8) + 12*y**2 + z*(5 - 6*z)) + 2*w2*y*(w1*(4*x - 9) + w3*(5 - 2*z)) + w3**2*(4*x**2 + 2*y**2 + z*(15 - 20*z))) + t*(2*w1*(2*x*z*(2 - 7*z) + y**2*(5 - 2*x)) - 2*w2*y*(x*(2*x - 9) + 4*y**2 + z*(5 - 6*z)) + w3*(x**2*(5 - 18*z) + y**2*(2*z - 5) + z**2*(22*z - 15))) + x**2*(2*y**2 + z*(14*z - 5)) - 10*x*y**2 + (y**2 + z**2)*(2*y**2 + z*(5 - 8*z)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + y**3 + (t*w1 - x)*(3*t*w2*y**2*(t*w1 - x) - t*w3 + y**3*(-4*t*w1 + 4*x) + z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (t*(w1**2 + w2**2 + w3**2) - w1*x - w2*y - w3*z)*(-2*t*w3*z**3 + x*z*(-t*w1 + x) + z**4 + z**2*(t**2*(w2**2 + w3**2) - 2*t*w2*y + y**2) + (t*w1 - x)*(t*w2*y + t*w3*x - y**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (-t**4*(w1*(w1*w3 + w2**2) + 2*w3*z*(w2**2 + w3**2))*(w1**2 + w2**2 + w3**2) + t**3*(w1**4*z + w1**3*(2*w2*y + 3*w3*x) + w1**2*(w2**2*(2*x + 2*z**2 + z) + w2*w3*y*(4*z + 1) + 2*w3**2*z*(3*z + 1)) + w1*(3*w2**3*y + w2**2*w3*(2*x*(z + 1) + z) + 2*w2*w3**2*y + 2*w3**3*x*(z + 1)) + (w2**2 + w3**2)*(w2**2*(x + 2*z**2) + 6*w2*w3*y*z + 8*w3**2*z**2)) - t**2*(w1**3*(3*x*z + y**2) + w1**2*(w2*y*(4*x + 5*z**2 + z) + 3*w3*x**2 + w3*z*(2*y**2 + 7*z**2 + z)) + w1*(w2**2*(x**2 + x*z*(z + 2) + 2*y**2) + w2*w3*y*(4*x*z + x + z) + w3**2*(x*z*(5*z + 3) + y**2)) + w2**3*(4*x*y + 6*y*z**2) + w2**2*w3*(2*x**2 + x*z + 6*y**2*z + 8*z**3) + w2*w3**2*y*(3*x + 14*z**2) + 2*w3**3*(x**2 + y**2*z + 6*z**3)) + t*(w1**2*(3*x**2*z + 2*x*y**2 + 3*z**2*(y**2 + z**2)) + w1*(-w2*y*(-2*x**2 - x*z*(4*z + 1) + y**2 + z**2) + w3*x*(x**2 + y**2*(2*z - 1) + 6*z**3)) + w2**2*(-x**2*z*(z - 2) + 5*x*y**2 + 2*z**2*(3*y**2 + z**2)) + w2*w3*y*(2*x**2 + 3*x*z + 2*z*(y**2 + 5*z**2)) + w3**2*(-x**2*z*(z - 4) + 2*x*y**2 + 4*z**2*(y**2 + 2*z**2))) + w1*(-x**3*z - x**2*y**2 - x*z*(y**2 + z**2)*(3*z - 1) + y**2*(y**2 + z**2)) - (w2*y + w3*z)*(-x**2*z*(z - 2) + 2*x*y**2 + 2*z**2*(y**2 + z**2)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t**4*(-w1**3*w2*y + w1**2*(-w2**2*z*(2*x + z) + w2*w3*y + w3**2*(-2*x*z + x)) + w1*w3*(w2**2*(x - 2*z**3 + z**2) - w2*w3*y + w3**2*z**2*(1 - 2*z)) - 2*w2*z*(w2**2 + w3**2)*(w2*x - w3*y)) + t**3*(w1**3*y**2 - w1**2*w3*(x*z*(2 - 5*z) + y**2) + w1*w3**2*(x**2*(4*z - 3) + y**2 + z**3*(7*z - 3)) + w2**3*y*z*(8*x - 3*z) - w2**2*(w1*y**2 + w1*z*(-4*x**2 - 2*x*z + x - 3*z**3 + z**2) + w3*x*(x - 2*z**2*(z + 1)) + 6*w3*y**2*z) + w2*y*(w1**2*(x*(4*z + 3) + z*(3*z - 1)) + 2*w1*w3*(-3*x + z**2*(2*z - 1) + z) + w3**2*(4*x*z + x - 7*z**2)) + 2*w3**3*z*(x*z*(z - 1) - y**2)) - t**2*(w1**2*(x*y**2*(2*z + 3) + x*z**2*(3*z - 1) + y**2*z*(2*z - 1)) + w1*w2*y*(x**2*(8*z + 3) + 6*x*z*(z - 1) - 2*y**2 + 6*z**4 - 2*z**3 + z**2) + w2**2*(2*x**3*z + x**2*z*(z - 1) + x*y**2*(12*z - 1) + 3*x*z**4 - 9*y**2*z**2) + w3**2*(2*x**3*(z - 1) + x*y**2*(2*z + 1) + x*z**3*(7*z - 6) - 7*y**2*z**2) + w3*(w1*(2*x**2*z*(5*z - 3) - 5*x*y**2 + y**2*z*(z*(2*z - 1) + 2) + z**4*(8*z - 3)) + w2*y*(-5*x**2 + 2*x*z*(2*z*(z + 1) + 1) - 6*y**2*z - 8*z**3))) + t*(w1*(x**2*(y**2*(4*z + 3) + 3*z**2*(2*z - 1)) + x*y**2*z*(4*z - 5) - y**4 + y**2*z**2*(z*(3*z - 1) + 1) + z**5*(3*z - 1)) + w2*y*(x**3*(4*z + 1) + x**2*z*(3*z - 5) + x*(y**2*(8*z - 2) + 6*z**4 + z**2) - 9*y**2*z**2 - 3*z**4) + w3*(x**3*z*(5*z - 4) - 4*x**2*y**2 + 2*x*z*(y**2*(z**2 + z + 1) + z**3*(4*z - 3)) - 2*y**2*z*(y**2 + 4*z**2))) - x**3*(y**2*(2*z + 1) + z**2*(3*z - 2)) - 2*x**2*y**2*z*(z - 2) + x*(y**4*(1 - 2*z) - y**2*(3*z**4 + z**2) + z**5*(2 - 3*z)) + 3*y**2*z**2*(y**2 + z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            -2*mu*(t**4*w1*w2*(w1**2 + w2**2 - w3**2) + t**3*(-w1**3*y + w1**2*w2*(4 - 3*x) + w1*(-3*w2**2*y + 2*w2*w3*(5*z - 2) + w3**2*y) - w2**3*x + w2*w3**2*(x + 4)) + t**2*(w1**2*y*(3*x - 5) + w1*w2*(x*(3*x - 8) + 3*y**2 + z*(4 - 10*z)) + 2*w1*w3*y*(2 - 5*z) + 3*w2**2*x*y + 5*w2*w3*x - 2*w2*w3*z*(5*x + 4) - w3**2*y*(x + 5)) + 2*t*z**2*(5*w1*y + w2*(5*x + 2)) + t*z*(-4*w1*y - 5*w2*x + 10*w3*y*(x + 1)) - t*(w1*y*(x*(3*x - 10) + y**2) + w2*x*(x*(x - 4) + 3*y**2) + 5*w3*x*y) + y*(5*x*z + x*(x*(x - 5) + y**2) - z**2*(10*x + 5)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + y**2*(-3*t*w1 + 3*x) + (t*w2 - y)*(3*t*w2*y**2*(t*w1 - x) - t*w3 + y**3*(-4*t*w1 + 4*x) + z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) - (t**2*(w2*(w1*z**2 - w3*x) + y*(w1**2 + w3**2)) - t*(w1*y*(2*x + z**2) + w2*x*z*(z - 1) - w3*y*(x - 2*z)) + y*(x**2 + x*z*(z - 1) + z**2))*(t*(w1**2 + w2**2 + w3**2) - w1*x - w2*y - w3*z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t**4*(w1**4*y + w1**3*w2*z*(2*x + z) + w1**2*w3*(-w2*x + 2*w2*z**3 + 2*w3*y) + w1*w2*(2*w2**2*x*z + w2*w3*y*(1 - 2*z) + w3**2*(x + z**2)) + w2**3*w3*z**2 + w2*w3**3*(-x + z**2) + w3**4*y) + t**3*(-2*w1**3*y*(x*(z + 2) + z**2) + w1**2*(w2*(y**2 + z*(-6*x**2 - 3*x*z + x - 3*z**3)) + 2*w3*y*(x - z*(z**2 + 2))) + w1*(w2**2*y*z*(-6*x + 3*z - 1) + 2*w2*w3*(x**2 + x*z*(-2*z**2 + z - 1) + y**2*(2*z - 1) - z**3) - w3**2*y*(5*x + 2*z**2)) - w2**3*(2*x**2*z + z**3) + w2**2*w3*y*(2*x*(z - 1) - 3*z**2) + w2*w3**2*(-2*x**2 - x*z*(z - 3) + y**2 - 3*z**3) + w3**3*y*(2*x - z*(z + 4))) + t**2*(w2**2*y*z*(6*x**2 + x*(2 - 3*z) + 3*z**2) + w2*(-w3*x**3 - x*y**2*(2*w1 - 4*w3) + 2*x*z**3*(-w1 + w3*x + w3) + z**4*(6*w1*x + w1 + 3*w3) + z**2*(x*(3*w1*x + w1 - w3*(2*x + 3)) + y**2*(-6*w1 + 3*w3)) + z*(2*x**2*(w1*(3*x - 1) + 2*w3) + 2*y**2*(3*w1*x + w1 - w3*(2*x + 1)))) + y*(w1**2*(6*x**2*(z + 1) + 2*x*z*(3*z - 1) - y**2 + 3*z**4 + 2*z**2) + w1*w3*(-4*x**2 + 2*x*z*(z*(2*z - 1) + 5) - 2*y**2*z + y**2 + 4*z**3) + w3**2*(4*x**2 + 2*x*z*(z - 3) - y**2 + 3*z**2*(z + 2)))) + t*(-w2*z**5 + x*y*(w2*x*y + x**2*(-4*w1 + 2*w3) + y**2*(2*w1 - 2*w3)) - z**4*(w1*y*(6*x + 2) + w2*x*(3*x + 1) + 3*w3*y) + z**3*(2*w1*x*y + w2*(2*x**2 + x - 3*y**2) - 2*w3*y*(x*(x + 2) + 2)) + z**2*(-w2*x**2*(x + 2) + w2*y**2*(6*x + 1) + x*y*(-w1*(6*x + 5) + 2*w3*(x + 3)) + y**3*(3*w1 - w3)) - z*(w2*x**3*(2*x - 1) + 2*w2*x*y**2*(3*x + 2) + 2*x**2*y*(w1*(3*x - 2) + 4*w3) + y**3*(2*w1*x + w1 - 2*w3*(x + 1)))) + y*(x**2*(x - y)*(x + y) + 2*x*z*(x**2*(x - 1) + y**2*(x + 1)) + z**5 + z**4*(x*(3*x + 2) + 1) + z**3*(-2*x*(x + 1) + y**2) + z**2*(2*x**2*(x + 2) - y**2*(3*x + 1))))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t**4*w2*(w1**2 + w2**2 + w3**2)*(w1**2 + w1*w3*(2*z - 1) + w3**2) + t**3*(-w1**4*y + w1**3*(w2*(-3*x - 2*z**2 + z) + w3*y*(1 - 2*z)) - w1**2*(-2*w2*w3*x + w2*w3*z*(4*x + 3) + y*(2*w2**2 + 2*w3**2)) + w1*(w2**3*(-2*x - 2*z**2 + z) + 2*w2**2*w3*y*(1 - 2*z) + w2*w3**2*(-3*x - 4*z**2 + 2*z) + w3**3*y*(1 - 2*z)) - w3*(w2**3*(-x + z*(2*x + 2)) + 2*w2**2*w3*y + w2*w3**2*(2*x*z - x + 3*z) + w3**3*y)) + t**2*(w1**3*y*(3*x + z*(3*z - 1)) + w1**2*(w2*(3*x**2 + x*z*(3*z - 2) + z**2) + w3*y*(x*(4*z - 3) + 3*z)) + w1*(w2**2*y*(5*x + 2*z*(2*z - 1)) + w2*w3*(y**2*(2*z - 1) + z*(2*x*(x + 2) + z*(z - 1))) + w3**2*y*(3*x + z*(5*z - 2))) + w2*z**2*(w2**2*(2*x + 1) + w3**2*(4*x + 3)) + x*(w2**2 + w3**2)*(w2*x - 2*w3*y) + z*(w2**2*w3*y*(4*x + 5) - w2*x*(w2**2 + w3**2) + w3**3*y*(2*x + 3))) + t*(w1**2*y*(-3*x**2 + 3*x*z + y**2 - z**2*(6*x + 1)) + w1*(-w2*(x**3 + x*(2*y**2 + z**2) - z*(-2*y**2*z + y**2 + z**3)) + w3*y*(2*x**2 - 2*x*z*(x + 2) - 3*z**3 + z**2)) - w2**2*y*(3*x**2 - 2*x*z + z**2*(4*x + 3)) - w2*w3*(x**3 + x**2*z + x*y**2*(2*z - 1) + x*z**2*(z + 1) + 2*y**2*z + z**3) + w3**2*y*(-x**2 + x*z*(4 - 5*z) + y**2 - 3*z**2)) + w1*x*y*(x**2 + x*z*(3*z - 2) - y**2 + z**2) + 2*w2*x**2*y**2 - w2*x*z**4 + z**3*(w2*x + w3*y*(3*x + 1)) + z**2*(-w2*x**3 + 2*w2*y**2*(x + 1) - 2*w3*x*y) + z*(x - y)*(x + y)*(w2*x + w3*y))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            mu*(-4*t**4*w1*w3*(w1**2 + w2**2) + t**3*(w1**3*(14*z - 3) + 12*w1**2*w3*x + w1*(w2**2*(14*z - 3) + 8*w2*w3*y + w3**2*(5 - 6*z)) + 4*w2**2*w3*(x + 2)) - t**2*(w1**2*x*(42*z - 11) + 2*w1*(w2*y*(14*z - 3) + w3*(6*x**2 + 2*y**2 + z*(5 - 7*z))) + w2**2*(14*x*z - 5*x + 8*z) + 2*w2*w3*y*(4*x + 9) + w3**2*x*(5 - 6*z)) + t*(w1*x**2*(42*z - 13) + w1*y**2*(14*z - 3) + w1*z**2*(5 - 8*z) + 2*w2*y*(x*(14*z - 5) + 9*z) + 2*w3*(2*x**3 + 2*x*y**2 + x*z*(5 - 7*z) + 5*y**2)) + 8*x*z**3 - 5*x*z**2 + 5*x*(x**2 + y**2) - z*(14*x**3 + 2*y**2*(7*x + 5)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 - (t*(w1**2 + w2**2 + w3**2) - w1*x - w2*y - w3*z)*(t**2*(w1**2*x + w1*w3*z**2 + w2*(w2*x - w3*y)) + t*(-w1*(2*x**2 + z**3) + w2*y*(-2*x + z) + w3*(-x*z**2 + y**2)) + x**3 + x*(y**2 + z**3) - y**2*z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (3*t**3*w1*w2*w3*y**2 + t**2*(w1**2 - w1*y**2*(3*w2*z + 4*w3*y) + w2*(w2 - 3*w3*x*y**2)) + t*(-2*w1*(x - 2*y**3*z) + w2*y*(3*x*y*z - 2) + 4*w3*x*y**3) + x**2 - 4*x*y**3*z + y**2)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (t**4*(-w1**3*(w2*y + w3*(-2*x*z + x)) - w1**2*(w2**2*z**2 + w2*w3*y + w3**2*z**2*(1 - 2*z)) - w1*w2*(w2**2*y + w2*w3*(-2*x*z + x + z**2) + 2*w3**2*y*z) + w2*(-w2*z**2*(w2**2 + w3**2) + w3**2*(w2*x - w3*y))) + t**3*(w1**3*(x*z*(1 - 3*z) + y**2) + w1**2*(w2*y*(4*x + 2*z**2 + z) + w3*(x**2*(4 - 6*z) + y**2 + z**3*(2 - 5*z))) + w1*(w2**2*(3*y**2 + z*(-x*z + x + z**2)) + 4*w2*w3*y*(-x*z + x + 2*z**2) + w3**2*z*(x*z*(3 - 4*z) + 2*y**2)) + 2*w2**3*y*(x + 2*z**2) + w2**2*w3*(-2*x**2*(z - 1) + x*z*(z - 2) - y**2 + 2*z**3) + w2*w3**2*y*(x*(2*z - 3) + z*(2*z + 3)) + w3**3*y**2) + t**2*(w1**2*(x**2*z*(9*z - 4) - 4*x*y**2 - y**2*z*(z + 1) + z**4*(3*z - 1)) + w1*(-w2*y*(5*x**2 - 2*x*z*(z - 2) + 3*y**2 + 6*z**3) + w3*(x**3*(6*z - 5) + x*y**2*(2*z - 3) + 2*x*z**3*(5*z - 3) - 7*y**2*z**2)) + w2**2*(2*x**2*z*(z - 1) + x*(-6*y**2 - z**3 + z**2) + y**2*z*(1 - 6*z) - z**4) + w2*w3*y*(x**2*(4*z - 5) + 2*x*z*(3 - 4*z) + 2*y**2 - z**2*(4*z + 3)) + w3**2*(2*x**2*z**2*(z - 1) - 2*x*y**2*(z - 1) - y**2*z*(z + 3))) + t*(w1*(x**3*z*(5 - 9*z) + 5*x**2*y**2 - x*y**2*z*(z - 3) + 3*x*z**4*(1 - 2*z) + y**4 + 5*y**2*z**3) + w2*y*(2*x**3 + x**2*z*(5 - 4*z) + 6*x*y**2 + 3*x*z**2*(2*z - 1) + y**2*z*(4*z - 2) + z**3*(2*z + 1)) + w3*(-2*x**4*(z - 1) + x**2*(y**2*(3 - 2*z) + z**3*(4 - 5*z)) + x*y**2*z*(7*z - 4) - y**4 + y**2*z**2*(2*z + 3))) + x**4*z*(3*z - 2) - 2*x**3*y**2 + x**2*z*(y**2*(2*z - 3) + z**3*(3*z - 2)) + x*y**2*(-2*y**2 + z**2*(2 - 5*z)) - y**2*z*(y**2*(z - 1) + z**2*(z + 1)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t**4*(w1**2 + w2**2 + w3**2)*(w1**3 + w1*(w2**2 + 2*w3**2*z) - w2**2*w3) + t**3*(-4*w1**4*x - w1**3*(3*w2*y + w3*z*(4*z + 1)) + w1**2*(w2**2*(-5*x + z) + 2*w2*w3*y - w3**2*x*(4*z + 3)) - w1*(3*w2**3*y + w2**2*w3*(-x + 4*z**2 + z) + 2*w2*w3**2*y*(z + 1) + 6*w3**3*z**2) + w2**4*(-x + z) + 3*w2**3*w3*y - w2**2*w3**2*(x + z*(2*x - 2)) + 2*w2*w3**3*y - 2*w3**4*x*z) + t**2*(w1**3*(6*x**2 + y**2 + 3*z**3) + w1**2*(w2*y*(7*x - 3*z) + w3*x*z*(7*z + 2) - w3*y**2) + w1*(w2**2*(4*x**2 - x*z + 3*y**2 + 3*z**3) + w2*w3*y*(-x + z*(3*z + 2)) + w3**2*(2*x**2*(z + 2) + y**2 + 6*z**3)) + w2**3*y*(3*x - 4*z) - w2**2*w3*(-4*x*z**2 + 2*y**2 + z**2) + w2*w3**2*y*(3*x + z*(2*x - 4)) - w3**3*(-6*x*z**2 + y**2)) + t*(w1**2*(-4*x**3 + x*(-2*y**2 - 6*z**3 + z**2) + 2*y**2*z) - w1*(w2*y*(5*x**2 - 3*x*z + y**2 + 3*z**3) + w3*z*(x**2*(2*z + 3) - y**2*z + y**2 + 2*z**3)) - w2**2*(x**3 + 3*x*y**2 + x*z**2*(3*z - 1) - 5*y**2*z) - w2*w3*y*(x**2 + x*z*(3*z + 2) + y**2 - 2*z**2) - 2*w3**2*(x**3 + x*(y**2 + 3*z**3) - y**2*z)) + 2*w3*x*z**4 + 3*x*z**3*(w1*x + w2*y) - z**2*(w2*x*y + w3*y**2*(x + 1) + x**2*(w1 + w3*x)) + z*(2*w3*x*(x**2 + y**2) - 2*y**2*(w1*x + w2*y)) + (x**2 + y**2)*(w1*x**2 + y*(w2*x + w3*y)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2
        ))

        coef_f_el = CoefficientFunction((
            (mu*(2*t**4*(w1**2*(w2**2 + 2*w3**2) + w2**2*(w2**2 + w3**2)) - t**3*(w1**2*(4*w2*y + 3*w3*(6*z - 1)) + 4*w1*(w2**2*(x - 2) + 2*w3**2*x) + 8*w2**3*y + w2**2*w3*(5 - 2*z) + 4*w2*w3**2*y + w3**3*(5 - 6*z)) + t**2*(w1**2*(2*y**2 + z*(14*z - 3)) + 4*w1*w3*x*(9*z - 2) + w2**2*(x*(2*x - 8) + 12*y**2 + z*(5 - 6*z)) + 2*w2*y*(w1*(4*x - 9) + w3*(5 - 2*z)) + w3**2*(4*x**2 + 2*y**2 + z*(15 - 20*z))) + t*(-4*w1*x*z*(7*z - 2) - 2*w1*y**2*(2*x - 5) - 2*w2*y*(x*(2*x - 9) + 4*y**2 + z*(5 - 6*z)) + w3*(x**2*(5 - 18*z) + y**2*(2*z - 5) + z**2*(22*z - 15))) + x**2*(2*y**2 + z*(14*z - 5)) - 10*x*y**2 + 2*y**4 - 6*y**2*z**2 + 5*y**2*z - 8*z**4 + 5*z**3) - (t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)*(-2*t*w3*z**3 + x*z*(-t*w1 + x) + z**4 + z**2*(t**2*(w2**2 + w3**2) - 2*t*w2*y + y**2) + (t*w1 - x)*(t*w2*y + t*w3*x - y**2)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            (-2*mu*(t**4*w1*w2*(w1**2 + w2**2 - w3**2) + t**3*(-w1**3*y + w1**2*w2*(4 - 3*x) + w1*(-3*w2**2*y + 2*w2*w3*(5*z - 2) + w3**2*y) - w2**3*x + w2*w3**2*(x + 4)) + t**2*(w1**2*y*(3*x - 5) + w1*w2*(x*(3*x - 8) + 3*y**2 + z*(4 - 10*z)) + 2*w1*w3*y*(2 - 5*z) + 3*w2**2*x*y + w2*w3*(5*x - z*(10*x + 8)) - w3**2*y*(x + 5)) - t*(w1*y*(x*(3*x - 10) + y**2 + z*(4 - 10*z)) + 5*w2*x*z + w2*x*(x*(x - 4) + 3*y**2) - 2*w2*z**2*(5*x + 2) + 5*w3*y*(x - z*(2*x + 2))) + y*(5*x*z + x*(x*(x - 5) + y**2) - z**2*(10*x + 5))) + (t**2*(w2*(w1*z**2 - w3*x) + y*(w1**2 + w3**2)) - t*(w1*y*(2*x + z**2) + w2*x*z*(z - 1) - w3*y*(x - 2*z)) + y*(x**2 + x*z*(z - 1) + z**2))*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            (-mu*(4*t**4*w1*w3*(w1**2 + w2**2) - t**3*(w1**3*(14*z - 3) + 12*w1**2*w3*x + w1*(w2**2*(14*z - 3) + 8*w2*w3*y + w3**2*(5 - 6*z)) + 4*w2**2*w3*(x + 2)) + t**2*(w1**2*x*(42*z - 11) + 2*w1*(w2*y*(14*z - 3) + w3*(6*x**2 + 2*y**2 + z*(5 - 7*z))) + w2**2*(14*x*z - 5*x + 8*z) + 2*w2*w3*y*(4*x + 9) + w3**2*x*(5 - 6*z)) - t*(w1*x**2*(42*z - 13) + w1*y**2*(14*z - 3) + w1*z**2*(5 - 8*z) + 2*w2*y*(x*(14*z - 5) + 9*z) + 2*w3*(2*x**3 + 2*x*y**2 + x*z*(5 - 7*z) + 5*y**2)) + x**3*(14*z - 5) + x*y**2*(14*z - 5) + x*z**2*(5 - 8*z) + 10*y**2*z) + (t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)*(t**2*(w1**2*x + w1*w3*z**2 + w2*(w2*x - w3*y)) + t*(-w1*(2*x**2 + z**3) + w2*y*(-2*x + z) + w3*(-x*z**2 + y**2)) + x**3 + x*(y**2 + z**3) - y**2*z))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2
        ))

        coef_g = CoefficientFunction((t**2*(w1**2 + w1*w3*(2*z - 1) + w3**2) + t*(w1*(-2*x - 4*z**2 + z) + 2*w2*y + 3*w3*x - 2*w3*z*(x + 1)) + x**2 + x*z*(4*z - 3) - 2*y**2 + z**2)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))
    else:
        wN = CoefficientFunction(
            (-x**2*(y**2 + z**2) - y**2*z**2)/(2*sqrt(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6))
        )

        coef_u = CoefficientFunction((
            -x*z*(t*(x**2 + y**2) + z**2)/sqrt(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6),
            y*z*(t*(x**2 + y**2) + z**2)/sqrt(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6),
            (x - y)*(x + y)*(t*z**2 + x**2 + y**2)/sqrt(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)
        ))

        coef_p = CoefficientFunction(
            y * (-t / 5 + x) + z
        )

        coef_g = CoefficientFunction(
            0.0
        )

        coef_f = CoefficientFunction((
            -x*(2*mu*z*(t**7*x**14*y**4 + 25*t**7*x**12*y**6 + 64*t**7*x**10*y**8 + 68*t**7*x**8*y**10 + 31*t**7*x**6*y**12 + 3*t**7*x**4*y**14 + 6*t**6*x**14*y**4 + 76*t**6*x**12*y**6 + 216*t**6*x**10*y**8 + 286*t**6*x**8*y**10 + 184*t**6*x**6*y**12 + 46*t**6*x**4*y**14 + 2*t**6*x**2*y**16 + t**5*x**16*y**2 + 50*t**5*x**14*y**4 + 82*t**5*x**12*y**6 + 161*t**5*x**10*y**8 + 355*t**5*x**8*y**10 + 406*t**5*x**6*y**12 + 202*t**5*x**4*y**14 + 23*t**5*x**2*y**16 + 8*t**4*x**16*y**2 + 22*t**4*x**14*y**4 + 54*t**4*x**12*y**6 + 96*t**4*x**10*y**8 + 84*t**4*x**8*y**10 + 156*t**4*x**6*y**12 + 224*t**4*x**4*y**14 + 102*t**4*x**2*y**16 + 6*t**4*y**18 + 14*t**3*x**16*y**2 - 3*t**3*x**14*y**4 + 81*t**3*x**12*y**6 + 34*t**3*x**10*y**8 - 130*t**3*x**8*y**10 - 173*t**3*x**6*y**12 - 77*t**3*x**4*y**14 + 46*t**3*x**2*y**16 + 16*t**3*y**18 + 4*t**2*x**18 + 4*t**2*x**16*y**2 + 62*t**2*x**14*y**4 + 16*t**2*x**12*y**6 - 28*t**2*x**10*y**8 - 98*t**2*x**8*y**10 - 204*t**2*x**6*y**12 - 170*t**2*x**4*y**14 - 82*t**2*x**2*y**16 + 4*t**2*z**18 + 9*t*x**16*y**2 + 12*t*x**14*y**4 + 8*t*x**12*y**6 - 39*t*x**10*y**8 - 49*t*x**8*y**10 - 52*t*x**6*y**12 - 84*t*x**4*y**14 - 49*t*x**2*y**16 - 12*t*y**18 + 4*t*z**16*(x**2*(2*t**3 + 5*t**2 + t + 1) + y**2*(t*(-2*t**2 + t + 4) + 6)) + t*z**12*(x**6*(t + 1)*(t*(t*(t*(t*(4*t + 25) + 62) + 54) + 6) + 17) + x**4*y**2*(t*(t*(t*(t*(t*(4*t - 9) - 61) + 304) + 392) + 333) + 45) + x**2*y**4*(t*(t*(-t*(t*(t*(4*t + 53) + 375) - 288) + 632) + 361) + 159) + y**6*(-t*(t*(t*(t*(t*(4*t + 31) + 107) + 4) - 252) - 59) + 171)) - 6*x**14*y**4 - 6*x**12*y**6 - 12*x**8*y**10 - 12*x**6*y**12 - 6*x**2*y**16 - 6*y**18 + z**14*(x**4*(t**2*(t + 2)*(t*(4*t*(t + 4) + 17) + 10) + 3) + x**2*y**2*(2*t*(t*(t*(78 - 31*t) + 82) + 12) + 6) + y**4*(t*(t*(-t*(2*t - 1)*(t*(2*t + 13) + 28) + 148) + 24) + 15)) + z**10*(x**8*(t*(t*(t*(t*(t*(t*(13*t + 71) + 146) + 137) + 75) + 49) + 10) + 3) + x**6*y**2*(t*(t*(t*(t*(2*t*(t*(10 - 7*t) + 97) + 647) + 838) + 202) + 138) - 9) + x**4*y**4*(2*t*(t*(t*(-t*(t*(t*(62*t + 139) + 100) - 641) + 814) + 277) + 72) + 18) + x**2*y**6*(t*(t*(t*(-2*t**2*(t*(57*t + 134) + 23) + 359*t + 1222) + 782) + 30) + 51) + y**8*(t*(t*(t*(t*(t*(-t*(17*t + 57) + 2) + 87) + 25) + 449) - 30) + 45)) + z**8*(x**10*(t*(t*(t*(t*(t*(t*(17*t + 79) + 131) + 121) + 103) + 37) + 13) + 3) + x**8*y**2*(t*(t*(t*(t*(t*(t*(t + 55) + 496) + 923) + 549) + 469) + 6) + 21) + x**6*y**4*(t*(t*(t*(t*(t*(-4*t*(50*t + 147) + 1165) + 2246) + 1562) + 708) + 117) + 30) + x**4*y**6*(t*(t*(t*(t*(t*(-4*t*(71*t + 200) + 1299) + 2096) + 1590) + 968) + 183) - 12) + x**2*y**8*(t*(t*(t*(t*(-3*t*(19*t*(t + 3) - 68) + 1109) + 719) + 471) + 290) - 45) + y**10*(t*(t*(t*(t*(11*t**2*(t + 3) + t + 105) + 325) - 157) + 207) - 21)) + z**6*(t*x**12*(t*(t*(3*t*(t*(t*(3*t + 13) + 25) + 32) + 71) + 29) + 17) + 2*t*x**10*y**2*(t*(t*(t*(t*(t*(t + 58) + 246) + 226) + 331) + 110) + 36) + t*x**4*y**8*(t*(t*(t*(t*(t*(543 - 85*t) + 1167) + 1916) + 1105) + 289) + 105) + x**8*y**4*(t*(t*(t*(t*(t*(t*(185 - 207*t) + 1033) + 1990) + 1507) + 391) + 111) + 30) + x**6*y**6*(2*t*(t*(t*(t*(t*(t*(133 - 194*t) + 926) + 1317) + 858) + 243) + 62) + 30) + x**2*y**10*(2*t*(t*(t*(t*(t*(t*(41*t + 189) + 270) + 239) + 255) + 31) - 32) + 30) + y**12*(t*(t*(t*(t*(t*(t*(11*t + 73) + 185) + 74) - 67) + 171) - 141) + 30)) + z**4*(t*x**6*y**8*(t*(t*(t*(t*(t*(139*t + 1287) + 1948) + 1012) + 521) + 181) - 48) + x**14*(t**2*(t + 2)*(t*(t*(t*(t + 4) + 20) + 5) + 17) + 3) + x**12*y**2*(t*(t*(t*(t*(t*(9*t*(t + 6) + 146) + 377) + 203) + 150) + 78) - 9) + x**10*y**4*(t*(t*(t*(t*(t*(t*(41*t + 37) + 874) + 982) + 563) + 467) + 78) - 18) + x**8*y**6*(t*(t*(t*(t*(t*(t*(17*t + 657) + 1468) + 1670) + 835) + 283) + 128) - 18) + x**4*y**10*(t*(t*(t*(t*(t*(t*(179*t + 1189) + 1354) + 562) + 65) - 205) - 138) + 18) + x**2*y**12*(t*(t*(t*(t*(t*(t*(59*t + 326) + 858) + 325) - 271) - 186) - 82) - 21) + y**14*(t*(t*(t*(t*(t*(t*(3*t + 28) + 116) + 171) - 7) - 164) + 24) - 27)) + z**2*(t*x**16*(t**2*(t*(t + 8) + 20) + 7) + 2*t*x**8*y**8*(t*(t*(t*(t*(t*(132*t + 383) + 610) + 96) + 112) - 59) - 14) + t*y**16*(t*(2*t + 7)*(t*(t*(t + 8) + 8) - 6) - 63) + x**14*y**2*(t**2*(t*(t*(2*t*(t + 2)*(t + 4) + 87) + 106) + 50) + 15) + x**12*y**4*(t*(t*(t*(t*(t*(t*(25*t + 91) + 193) + 189) + 391) + 89) + 15) + 15) + x**10*y**6*(t*(t*(t*(t*(2*t*(32*t*(2*t + 7) + 127) + 809) + 296) + 62) - 2) + 21) + x**6*y**10*(t*(t*(t*(t*(2*t*(t*(100*t + 533) + 569) + 527) - 628) - 200) - 66) - 21) + x**4*y**12*(t*(t*(t*(t*(t*(t*(79*t + 437) + 1171) + 307) - 439) - 457) - 75) - 15) + x**2*y**14*(t*(t*(t*(t*(2*t*(t*(3*t + 29) + 184) + 433) - 206) - 280) - 76) - 15))) - z*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)*(2*t**4*x**8*y**6 + 2*t**4*x**6*y**8 + 2*t**3*x**10*y**4 + 6*t**3*x**8*y**6 + 2*t**3*x**6*y**8 - 4*t**3*x**4*y**10 - 2*t**3*x**2*y**12 + t**2*x**12*y**2 + 7*t**2*x**10*y**4 - 12*t**2*x**6*y**8 - 11*t**2*x**4*y**10 - 5*t**2*x**2*y**12 + t*x**12*y**2 - t*x**10*y**4 - 4*t*x**8*y**6 - 10*t*x**6*y**8 - 13*t*x**4*y**10 - 7*t*x**2*y**12 - 2*t*y**14 + 2*t*z**12*(x**2 + 3*y**2) - 2*x**10*y**4 - 4*x**8*y**6 - 2*x**6*y**8 - 2*x**4*y**10 - 4*x**2*y**12 - 2*y**14 + z**10*(x**4*(t**2*(t + 5) + 4) + x**2*y**2*(2*t*(t*(3*t + 11) + 4) + 6) + y**4*(t*(t*(5*t + 17) + 2) + 2)) + z**8*(t*x**6*(t*(t*(t + 2) + 3) + 14) + t*y**6*(t*(t*(5*t + 16) + 13) + 6) + x**4*y**2*(t + 1)*(t*(t*(7*t + 16) + 26) + 6) + x**2*y**4*(t*(t*(t*(11*t + 53) + 40) + 24) + 2)) + z**6*(x**8*(t*(t*(-t*(t - 3) + 12) + 3) + 3) + x**6*y**2*(t*(t*(t*(3*t + 31) + 68) + 35) + 3) + x**4*y**4*(2*t*(4*t*(2*t*(2*t + 5) + 11) + 21) - 2) + x**2*y**6*(t*(t*(t*(31*t + 47) + 44) + 19) - 1) + y**8*(t + 1)*(3*t**2*(t + 2) + 1)) + z**4*(x**10*(t*(t*(2*t + 3) + 4) + 1) + x**8*y**2*(2*t*(t*(t*(16 - 3*t) + 18) + 11) + 6) + x**6*y**4*(t*(t*(t*(8*t + 77) + 92) + 21) + 2) + x**4*y**6*(t*(t*(t*(32*t + 51) + 58) + 15) + 4) + x**2*y**8*(2*t*(t*(t*(t + 8) + 4) - 1) + 6) - y**10*(t*(t + 2)*(4*t - 1) + 1)) + z**2*(t*x**12*(t + 1) + 2*t*x**6*y**6*(t*(t*(7*t + 10) + 9) + 4) - t*y**12*(t*(2*t + 5) + 7) + x**10*y**2*(t*(t*(4*t + 11) + 8) + 3) + x**8*y**4*(t*(t*(t*(35 - 3*t) + 28) + 8) + 2) + x**4*y**8*(t**2*(t*(t + 9) - 26) + 6) - x**2*y**10*(t*(t*(16*t + 17) + 4) + 1))) - (t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**(3/2)*(2*t*y**2*(t + 1)*(-x + y)*(x + y)*(x**2 + y**2)**3*(t*x**2 + y**2) + 2*t*z**10*(x**2*(t + 4) + y**2*(3*t + 4)) + 2*t*z**8*(x**4*(2*t*(t + 3) + 2) + x**2*y**2*(5*t*(2*t + 3) + 1) + y**4*(t*(4*t + 7) + 9)) + 2*t*z**4*(x**2 + y**2)*(x**6*(t**2*(t + 2) + 2) + x**4*y**2*(t*(t*(11*t + 7) - 6) - 1) + x**2*y**4*(t*(t*(5*t + 13) + 14) - 1) + y**6*(-t*(t - 4)*(t + 2) + 16)) + 2*z**12 + 2*z**6*(t*x**4*y**2*(t*(t*(11*t + 17) + 5) + 1) + x**6*(t**2*(t + 2)**2 + 1) + x**2*y**4*(t*(t + 2)*(t*(7*t + 8) + 2) + 3) + y**6*(t*(t*(t*(5 - 3*t) + 21) + 3) + 4)) + 2*z**2*(x**2 + y**2)**2*(t**2*x**6 + t*x**4*y**2*(t*(t**2 + t - 2) - 3) - x**2*y**4*(t**2*(t*(t - 6) - 5) + 3) + y**6*(t*(t + 1)*(t + 3) + 3))))/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**(7/2)) + (-5*t*x*z**5 - 5*t*x*z*(x**2 + y**2)*(t*y**2 + x**2) + t*y*z**4*(x*(t**2 - x*(5*t - 10)) + y**2*(5*t + 10)) + t*y*z**2*(x**3*(t**2 + t - 5*x) + x*y**2*(t**2 + 15*t*x + t - 5*x) + y**4*(5*t + 10)) - 5*x*z**3*(t*y**2*(t + 1) + x**2*(t**2 + 1)) + 5*y*z**6 + y*(t*x**2 + y**2)*(t**2*x*y**2 + x**3*(t - 5*x) + 5*y**4))/(5*t*x**4*(t + 2)*(y**2 + z**2) + 5*t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + 5*t*y**4*z**2*(t + 2) + 5*t*y**2*z**4*(t + 2) + 5*x**6 + 5*y**6 + 5*z**6),
            y*(2*mu*z*(x**18*(6*t**4 + 16*t**3 - 12*t - 6) + x**16*(t*z**2*(t*(2*t + 7)*(t*(t*(t + 8) + 8) - 6) - 63) + y**2*(t*(t*(t*(t*(t*(2*t + 23) + 102) + 46) - 82) - 49) - 6)) + x**14*(t*y**4*(t*(t*(t*(3*t + 16)*(t*(t + 10) + 14) - 77) - 170) - 84) + y**2*z**2*(t*(t*(t*(t*(2*t*(t*(3*t + 29) + 184) + 433) - 206) - 280) - 76) - 15) + z**4*(t*(t*(t*(t*(t*(t*(3*t + 28) + 116) + 171) - 7) - 164) + 24) - 27)) + x**12*(y**6*(t*(t*(t*(t*(t*(t*(31*t + 184) + 406) + 156) - 173) - 204) - 52) - 12) + y**4*z**2*(t*(t*(t*(t*(t*(t*(79*t + 437) + 1171) + 307) - 439) - 457) - 75) - 15) + y**2*z**4*(t*(t*(t*(t*(t*(t*(59*t + 326) + 858) + 325) - 271) - 186) - 82) - 21) + z**6*(t*(t*(t*(t*(t*(t*(11*t + 73) + 185) + 74) - 67) + 171) - 141) + 30)) + x**10*(y**8*(t*(t*(t*(t*(t*(68*t**2 + 286*t + 355) + 84) - 130) - 98) - 49) - 12) + y**6*z**2*(t*(t*(t*(t*(2*t*(t*(100*t + 533) + 569) + 527) - 628) - 200) - 66) - 21) + y**4*z**4*(t*(t*(t*(t*(t*(t*(179*t + 1189) + 1354) + 562) + 65) - 205) - 138) + 18) + y**2*z**6*(2*t*(t*(t*(t*(t*(t*(41*t + 189) + 270) + 239) + 255) + 31) - 32) + 30) + z**8*(t*(t*(t*(t*(11*t**2*(t + 3) + t + 105) + 325) - 157) + 207) - 21)) + x**8*(t*y**10*(t*(t*(t*(t*(8*t*(8*t + 27) + 161) + 96) + 34) - 28) - 39) + 2*t*y**8*z**2*(t*(t*(t*(t*(t*(132*t + 383) + 610) + 96) + 112) - 59) - 14) + t*y**6*z**4*(t*(t*(t*(t*(t*(139*t + 1287) + 1948) + 1012) + 521) + 181) - 48) + t*y**4*z**6*(t*(t*(t*(t*(t*(543 - 85*t) + 1167) + 1916) + 1105) + 289) + 105) + y**2*z**8*(t*(t*(t*(t*(-3*t*(19*t*(t + 3) - 68) + 1109) + 719) + 471) + 290) - 45) + z**10*(t*(t*(t*(t*(t*(-t*(17*t + 57) + 2) + 87) + 25) + 449) - 30) + 45)) + x**6*(t*z**12*(-t*(t*(t*(t*(t*(4*t + 31) + 107) + 4) - 252) - 59) + 171) + y**12*(t*(t*(t*(t*(t*(t*(25*t + 76) + 82) + 54) + 81) + 16) + 8) - 6) + y**10*z**2*(t*(t*(t*(t*(2*t*(32*t*(2*t + 7) + 127) + 809) + 296) + 62) - 2) + 21) + y**8*z**4*(t*(t*(t*(t*(t*(t*(17*t + 657) + 1468) + 1670) + 835) + 283) + 128) - 18) + y**6*z**6*(2*t*(t*(t*(t*(t*(t*(133 - 194*t) + 926) + 1317) + 858) + 243) + 62) + 30) + y**4*z**8*(t*(t*(t*(t*(t*(-4*t*(71*t + 200) + 1299) + 2096) + 1590) + 968) + 183) - 12) + y**2*z**10*(t*(t*(t*(-2*t**2*(t*(57*t + 134) + 23) + 359*t + 1222) + 782) + 30) + 51)) + x**4*(t*y**2*z**12*(t*(t*(-t*(t*(t*(4*t + 53) + 375) - 288) + 632) + 361) + 159) + y**14*(t*(t*(t*(t*(t*(t*(t + 6) + 50) + 22) - 3) + 62) + 12) - 6) + y**12*z**2*(t*(t*(t*(t*(t*(t*(25*t + 91) + 193) + 189) + 391) + 89) + 15) + 15) + y**10*z**4*(t*(t*(t*(t*(t*(t*(41*t + 37) + 874) + 982) + 563) + 467) + 78) - 18) + y**8*z**6*(t*(t*(t*(t*(t*(t*(185 - 207*t) + 1033) + 1990) + 1507) + 391) + 111) + 30) + y**6*z**8*(t*(t*(t*(t*(t*(-4*t*(50*t + 147) + 1165) + 2246) + 1562) + 708) + 117) + 30) + y**4*z**10*(2*t*(t*(t*(-t*(t*(t*(62*t + 139) + 100) - 641) + 814) + 277) + 72) + 18) + z**14*(t*(t*(-t*(2*t - 1)*(t*(2*t + 13) + 28) + 148) + 24) + 15)) + x**2*(t*y**16*(t*(t + 2)*(t*(t + 6) + 2) + 9) + 2*t*y**10*z**6*(t*(t*(t*(t*(t*(t + 58) + 246) + 226) + 331) + 110) + 36) + t*y**4*z**12*(t*(t*(t*(t*(t*(4*t - 9) - 61) + 304) + 392) + 333) + 45) + 4*t*z**16*(t*(-2*t**2 + t + 4) + 6) + y**14*z**2*(t**2*(t*(t*(2*t*(t + 2)*(t + 4) + 87) + 106) + 50) + 15) + y**12*z**4*(t*(t*(t*(t*(t*(9*t*(t + 6) + 146) + 377) + 203) + 150) + 78) - 9) + y**8*z**8*(t*(t*(t*(t*(t*(t*(t + 55) + 496) + 923) + 549) + 469) + 6) + 21) + y**6*z**10*(t*(t*(t*(t*(2*t*(t*(10 - 7*t) + 97) + 647) + 838) + 202) + 138) - 9) + y**2*z**14*(2*t*(t*(t*(78 - 31*t) + 82) + 12) + 6)) + (y**2 + z**2)*(t*y**2 + z**2)*(y**4 + y**2*z**2*(t*(t + 2) - 1) + z**4)*(4*t**2*z**10 + 4*t*y**10 + 8*t*y**6*z**4*(t*(t + 1)**2 + 1) + 4*t*y**2*z**8*(t*(t + 1)**2 + 1) + y**8*z**2*(t**2*(t*(t + 4) + 12) + 3) + y**4*z**6*(t**2*(t*(9*t + 20) + 8) + 3))) + z*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)*(x**14*(2*t + 2) + x**12*(t*z**2*(t*(2*t + 5) + 7) + y**2*(t + 1)*(t*(2*t + 3) + 4)) + x**10*(y**4*(t*(t*(4*t + 11) + 13) + 2) + y**2*z**2*(t*(t*(16*t + 17) + 4) + 1) + z**4*(t*(t + 2)*(4*t - 1) + 1)) + x**8*(y**6*(-2*t*(t - 2) + 2)*(t*(t + 3) + 1) - y**4*z**2*(t**2*(t*(t + 9) - 26) + 6) - y**2*z**4*(2*t*(t*(t*(t + 8) + 4) - 1) + 6) - z**6*(t + 1)*(3*t**2*(t + 2) + 1)) + x**6*(-2*t*y**6*z**2*(t*(t*(7*t + 10) + 9) + 4) - t*z**8*(t*(t*(5*t + 16) + 13) + 6) + y**8*(-2*t**4 - 6*t**3 + 4*t + 4) - y**4*z**4*(t*(t*(t*(32*t + 51) + 58) + 15) + 4) - y**2*z**6*(t*(t*(t*(31*t + 47) + 44) + 19) - 1)) + x**4*(y**10*(-2*t**3 - 7*t**2 + t + 2) + y**8*z**2*(t*(t*(t*(3*t - 35) - 28) - 8) - 2) - y**6*z**4*(t*(t*(t*(8*t + 77) + 92) + 21) + 2) - y**4*z**6*(2*t*(4*t*(2*t*(2*t + 5) + 11) + 21) - 2) - y**2*z**8*(t*(t*(t*(11*t + 53) + 40) + 24) + 2) - z**10*(t*(t*(5*t + 17) + 2) + 2)) + x**2*(-t*y**12*(t + 1) - 6*t*z**12 - y**10*z**2*(t*(t*(4*t + 11) + 8) + 3) + y**8*z**4*(2*t*(t*(t*(3*t - 16) - 18) - 11) - 6) - y**6*z**6*(t*(t*(t*(3*t + 31) + 68) + 35) + 3) - y**4*z**8*(t + 1)*(t*(t*(7*t + 16) + 26) + 6) - y**2*z**10*(2*t*(t*(3*t + 11) + 4) + 6)) + y**2*z**2*(-t*y**10*(t + 1) - t*y**4*z**6*(t*(t*(t + 2) + 3) + 14) - 2*t*z**10 - y**8*z**2*(t*(t*(2*t + 3) + 4) + 1) + y**6*z**4*(t*(t*(t*(t - 3) - 12) - 3) - 3) - y**2*z**8*(t**2*(t + 5) + 4))) + (t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**(3/2)*(2*t*x**2*(t + 1)*(x - y)*(x + y)*(x**2 + y**2)**3*(t*y**2 + x**2) + 2*t*z**10*(x**2*(3*t + 4) + y**2*(t + 4)) + 2*t*z**8*(x**4*(t*(4*t + 7) + 9) + x**2*y**2*(5*t*(2*t + 3) + 1) + y**4*(2*t*(t + 3) + 2)) + 2*t*z**4*(x**2 + y**2)*(x**6*(-t*(t - 4)*(t + 2) + 16) + x**4*y**2*(t*(t*(5*t + 13) + 14) - 1) + x**2*y**4*(t*(t*(11*t + 7) - 6) - 1) + y**6*(t**2*(t + 2) + 2)) + 2*z**12 + 2*z**6*(t*x**2*y**4*(t*(t*(11*t + 17) + 5) + 1) + x**6*(t*(t*(t*(5 - 3*t) + 21) + 3) + 4) + x**4*y**2*(t*(t + 2)*(t*(7*t + 8) + 2) + 3) + y**6*(t**2*(t + 2)**2 + 1)) + 2*z**2*(x**2 + y**2)**2*(t**2*y**6 + t*x**2*y**4*(t*(t**2 + t - 2) - 3) + x**6*(t*(t + 1)*(t + 3) + 3) - x**4*y**2*(t**2*(t*(t - 6) - 5) + 3))))/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**(7/2)) - (y*(t*x*(y**2 + z**2) + x**3)*(t*y*(x**2 + z**2) + y**3) + (t - 5*x)*(t*x**4*(2*y**2 + z**2*(t + 2)) + t*x**2*(t*y**4 + 4*t*y**2*z**2 + z**4*(t + 2)) + x**6 + (t*y**2*z + z**3)**2)/5 + (t*y*(x**2 + z**2) + y**3)*(t*z*(x**2 + y**2) + z**3))/(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6),
            (2*mu*(x - y)*(x + y)*(-2*t**7*x**12*y**6 - 6*t**7*x**10*y**8 - 6*t**7*x**8*y**10 - 2*t**7*x**6*y**12 - t**6*x**14*y**4 - 15*t**6*x**12*y**6 - 48*t**6*x**10*y**8 - 48*t**6*x**8*y**10 - 15*t**6*x**6*y**12 - t**6*x**4*y**14 - 16*t**5*x**14*y**4 - 78*t**5*x**12*y**6 - 170*t**5*x**10*y**8 - 170*t**5*x**8*y**10 - 78*t**5*x**6*y**12 - 16*t**5*x**4*y**14 - 5*t**4*x**16*y**2 - 44*t**4*x**14*y**4 - 148*t**4*x**12*y**6 - 187*t**4*x**10*y**8 - 187*t**4*x**8*y**10 - 148*t**4*x**6*y**12 - 44*t**4*x**4*y**14 - 5*t**4*x**2*y**16 - 8*t**3*x**16*y**2 - 16*t**3*x**14*y**4 + 18*t**3*x**12*y**6 + 110*t**3*x**10*y**8 + 110*t**3*x**8*y**10 + 18*t**3*x**6*y**12 - 16*t**3*x**4*y**14 - 8*t**3*x**2*y**16 + 4*t**3*z**18 + 18*t**2*x**16*y**2 + 67*t**2*x**14*y**4 + 285*t**2*x**12*y**6 + 398*t**2*x**10*y**8 + 398*t**2*x**8*y**10 + 285*t**2*x**6*y**12 + 67*t**2*x**4*y**14 + 18*t**2*x**2*y**16 + 4*t**2*z**16*(t + 2)*(2*t**2 + 1)*(x**2 + y**2) + 4*t*x**18 + 16*t*x**16*y**2 + 124*t*x**14*y**4 + 210*t*x**12*y**6 + 326*t*x**10*y**8 + 326*t*x**8*y**10 + 210*t*x**6*y**12 + 124*t*x**4*y**14 + 16*t*x**2*y**16 + 4*t*y**18 + t*z**14*(x**4*(t*(t*(t*(t*(4*t*(t + 4) + 33) + 48) + 32) + 4) + 7) + x**2*y**2*(2*t**2*(t*(t*(2*t + 1)*(2*t + 7) + 90) + 16) + 22) + y**4*(t*(t*(t*(t*(4*t*(t + 4) + 33) + 48) + 32) + 4) + 7)) + t*z**10*(x**8*(t*(t*(t*(t*(t*(17*t + 83) + 147) + 128) + 79) + 33) + 17) + x**6*y**2*(t*(t*(t*(t*(4*t*(5*t + 37) + 627) + 688) + 166) + 368) - 1) + x**4*y**4*(2*t*(t*(t*(t*(t*(33 - 13*t) + 618) + 422) + 149) + 329) - 52) + x**2*y**6*(t*(t*(t*(t*(4*t*(5*t + 37) + 627) + 688) + 166) + 368) - 1) + y**8*(t*(t*(t*(t*(t*(17*t + 83) + 147) + 128) + 79) + 33) + 17)) + t*z**4*(x**2 + y**2)*(x**12*(t*(t*(t*(t*(t + 7) + 36) + 64) + 19) + 17) + x**10*y**2*(4*t*(-t*(t*(t*(t + 12) - 2) - 114) + 104) + 36) + x**8*y**4*(t*(t*(-t*(t*(t*(31*t + 193) + 806) - 820) + 1683) + 517) + 170) - x**6*y**6*(2*t*(t*(t*(t*(t*(29*t + 482) + 63) - 612) - 781) - 554) - 134) + x**4*y**8*(t*(t*(-t*(t*(t*(31*t + 193) + 806) - 820) + 1683) + 517) + 170) + x**2*y**10*(4*t*(-t*(t*(t*(t + 12) - 2) - 114) + 104) + 36) + y**12*(t*(t*(t*(t*(t + 7) + 36) + 64) + 19) + 17)) + 15*x**16*y**2 + 30*x**14*y**4 + 66*x**12*y**6 + 81*x**10*y**8 + 81*x**8*y**10 + 66*x**6*y**12 + 30*x**4*y**14 + 15*x**2*y**16 + z**12*(x**2 + y**2)*(x**4*(t**2*(t*(t*(t*(t*(13*t + 58) + 92) + 77) + 63) + 30) + 3) + x**2*y**2*(2*t*(t*(t*(t*(t*(-t*(t - 40) + 110) + 31) + 151) - 4) + 6) + 6) + y**4*(t**2*(t*(t*(t*(t*(13*t + 58) + 92) + 77) + 63) + 30) + 3)) + z**8*(x**2 + y**2)*(x**8*(t*(t*(t*(t*(t*(t*(9*t + 55) + 139) + 161) + 79) + 45) + 13) + 3) + x**6*y**2*(t*(t*(t*(t*(2*t*(t*(16 - 6*t) + 187) + 755) + 696) + 42) + 138) - 9) + x**4*y**4*(2*t*(t*(t*(t*(t*(-3*t*(35*t + 9) + 323) + 798) + 357) + 77) + 101) - 24) + x**2*y**6*(t*(t*(t*(t*(2*t*(t*(16 - 6*t) + 187) + 755) + 696) + 42) + 138) - 9) + y**8*(t*(t*(t*(t*(t*(t*(9*t + 55) + 139) + 161) + 79) + 45) + 13) + 3)) + z**6*(x**12*(t*(t*(t*(t*(t*(t*(t + 15) + 66) + 113) + 87) + 41) + 10) + 3) + x**10*y**2*(2*t*(t*(t*(t*(t*(t*(2*t - 3) + 20) + 384) + 396) + 189) + 8) + 24) + x**8*y**4*(t*(t*(t*(t*(t*(-t*(97*t + 723) + 340) + 2405) + 2205) + 807) + 28) + 75) + x**6*y**6*(4*t*(t*(t*(t*(t*(-t*(70*t + 327) + 217) + 835) + 726) + 267) + 5) + 108) + x**4*y**8*(t*(t*(t*(t*(t*(-t*(97*t + 723) + 340) + 2405) + 2205) + 807) + 28) + 75) + x**2*y**10*(2*t*(t*(t*(t*(t*(t*(2*t - 3) + 20) + 384) + 396) + 189) + 8) + 24) + y**12*(t*(t*(t*(t*(t*(t*(t + 15) + 66) + 113) + 87) + 41) + 10) + 3)) + z**2*(x**16*(t**2*(t*(t + 8) + 24) + 3) + x**14*y**2*(t*(t*(t*(-t*(9*t + 46) + 66) + 160) + 123) - 6) + x**12*y**4*(t*(-t*(t*(t*(t*(t*(3*t + 17) + 235) + 239) - 427) - 853) + 219) + 3) + x**10*y**6*(t*(-t*(t*(t*(t*(32*t*(t + 6) + 703) + 386) - 1534) - 1360) + 429) + 6) - x**8*y**8*(2*t*(t*(t*(t*(3*t*(t*(7*t + 69) + 167) - 62) - 741) - 931) - 261) + 12) + x**6*y**10*(t*(-t*(t*(t*(t*(32*t*(t + 6) + 703) + 386) - 1534) - 1360) + 429) + 6) + x**4*y**12*(t*(-t*(t*(t*(t*(t*(3*t + 17) + 235) + 239) - 427) - 853) + 219) + 3) + x**2*y**14*(t*(t*(t*(-t*(9*t + 46) + 66) + 160) + 123) - 6) + y**16*(t**2*(t*(t + 8) + 24) + 3))) - (x - y)*(x + y)*(2*t**2*z**12*(x**2 + y**2) + t*z**10*(x**4*(t**2*(t + 3) + 6) + x**2*y**2*(2*t**3 + 6*t**2 + 2*t + 16) + y**4*(t**2*(t + 3) + 6)) + x**2*y**2*(x**10*(3*t + 3) + x**8*y**2*(t*(2*t*(t + 4) + 15) + 5) + x**6*y**4*(2*t*(t + 2)*(4*t + 5) + 6) + x**4*y**6*(2*t*(t + 2)*(4*t + 5) + 6) + x**2*y**8*(t*(2*t*(t + 4) + 15) + 5) + y**10*(3*t + 3)) - z**8*(x**2 + y**2)*(x**4*(t**2*(t*(t - 4) - 13) - 4) + x**2*y**2*(t*(3*t*(t*(t - 7) - 10) - 16) - 6) + y**4*(t**2*(t*(t - 4) - 13) - 4)) + z**6*(t*x**8*(t*(t + 6) + 13) + t*y**8*(t*(t + 6) + 13) + x**6*y**2*(t*(t*(t*(33 - 10*t) + 74) + 37) + 6) + x**4*y**4*(2*t*(t*(t*(43 - 11*t) + 58) + 26) + 8) + x**2*y**6*(t*(t*(t*(33 - 10*t) + 74) + 37) + 6)) + z**4*(x**2 + y**2)*(t*x**4*y**4*(t*(t*(69 - 11*t) + 85) + 37) + x**8*(t*(3*t + 4) + 3) + x**6*y**2*(t*(t*(6*t + 53) + 38) + 3) + x**2*y**6*(t*(t*(6*t + 53) + 38) + 3) + y**8*(t*(3*t + 4) + 3)) + z**2*(x**12*(t + 1) + x**10*y**2*(t*(t*(2*t + 11) + 22) + 7) + x**8*y**4*(t*(3*t*(7*t + 25) + 44) + 10) + x**6*y**6*(2*t*(t*(-t*(t - 26) + 53) + 28) + 8) + x**4*y**8*(t*(3*t*(7*t + 25) + 44) + 10) + x**2*y**10*(t*(t*(2*t + 11) + 22) + 7) + y**12*(t + 1)))*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6) - (2*t*z**11*(x**2 + y**2) + 2*t*z**7*(x**2 + y**2)*(2*t*x**2*y**2*(t*(3*t + 7) + 8) + x**4*(t*(t*(t + 3) + 3) + 3) + y**4*(t*(t*(t + 3) + 3) + 3)) + 2*t*z**5*(x**8*(t*(t*(t + 3) + 3) + 3) + x**6*y**2*(t + 1)*(t*(8*t + 19) + 5) + x**4*y**4*(6*t*(t*(5*t + 6) + 3) + 24) + x**2*y**6*(t + 1)*(t*(8*t + 19) + 5) + y**8*(t*(t*(t + 3) + 3) + 3)) + 2*t*z*(x**2 + y**2)**2*(x**8 + x**6*y**2*(t*(t + 6) + 1) + x**4*y**4*(2*t*(3*t + 2) + 4) + x**2*y**6*(t*(t + 6) + 1) + y**8) + 2*z**9*(2*t*x**2*y**2*(3*t*(t + 1) + 1) + x**4*(t**2*(t + 3) + 1) + y**4*(t**2*(t + 3) + 1)) + 2*z**3*(x**2 + y**2)*(x**8*(t**2*(t + 3) + 1) + x**6*y**2*(t*(t*(t*(t + 16) + 14) + 6) - 1) + x**4*y**4*(2*t*(t*(t*(7*t + 11) + 7) + 2) + 8) + x**2*y**6*(t*(t*(t*(t + 16) + 14) + 6) - 1) + y**8*(t**2*(t + 3) + 1)))*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**(3/2))/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**(7/2)) + (5*t**2*z**4*(x**2 + y**2) + 5*t*x**4*y**2*(t + 2) + 5*t*x**2*y**4*(t + 2) + t*y*z**5*(t - 10*x) + t*y*z*(x**2 + y**2)*(x**2*(t**2 - x*(5*t + 5)) + y**2*(t - x*(5*t + 5))) + 10*t*z**2*(2*t*x**2*y**2 + x**4 + y**4) + 5*x**6 + 5*y**6 + y*z**3*(x**2*(t**2*(t + 1) - x*(10*t**2 + 5*t + 5)) + y**2*(t**3 + t - x*(10*t**2 + 5*t + 5))))/(5*t*x**4*(t + 2)*(y**2 + z**2) + 5*t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + 5*t*y**4*z**2*(t + 2) + 5*t*y**2*z**4*(t + 2) + 5*x**6 + 5*y**6 + 5*z**6)
        ))

        coef_f_el = CoefficientFunction((
            x*z*(mu*(-t**7*x**14*y**4 - 25*t**7*x**12*y**6 - 64*t**7*x**10*y**8 - 68*t**7*x**8*y**10 - 31*t**7*x**6*y**12 - 3*t**7*x**4*y**14 - 6*t**6*x**14*y**4 - 76*t**6*x**12*y**6 - 216*t**6*x**10*y**8 - 286*t**6*x**8*y**10 - 184*t**6*x**6*y**12 - 46*t**6*x**4*y**14 - 2*t**6*x**2*y**16 - t**5*x**16*y**2 - 50*t**5*x**14*y**4 - 82*t**5*x**12*y**6 - 161*t**5*x**10*y**8 - 355*t**5*x**8*y**10 - 406*t**5*x**6*y**12 - 202*t**5*x**4*y**14 - 23*t**5*x**2*y**16 - 8*t**4*x**16*y**2 - 22*t**4*x**14*y**4 - 54*t**4*x**12*y**6 - 96*t**4*x**10*y**8 - 84*t**4*x**8*y**10 - 156*t**4*x**6*y**12 - 224*t**4*x**4*y**14 - 102*t**4*x**2*y**16 - 6*t**4*y**18 - 14*t**3*x**16*y**2 + 3*t**3*x**14*y**4 - 81*t**3*x**12*y**6 - 34*t**3*x**10*y**8 + 130*t**3*x**8*y**10 + 173*t**3*x**6*y**12 + 77*t**3*x**4*y**14 - 46*t**3*x**2*y**16 - 16*t**3*y**18 - 4*t**2*x**18 - 4*t**2*x**16*y**2 - 62*t**2*x**14*y**4 - 16*t**2*x**12*y**6 + 28*t**2*x**10*y**8 + 98*t**2*x**8*y**10 + 204*t**2*x**6*y**12 + 170*t**2*x**4*y**14 + 82*t**2*x**2*y**16 - 4*t**2*z**18 - 9*t*x**16*y**2 - 12*t*x**14*y**4 - 8*t*x**12*y**6 + 39*t*x**10*y**8 + 49*t*x**8*y**10 + 52*t*x**6*y**12 + 84*t*x**4*y**14 + 49*t*x**2*y**16 + 12*t*y**18 - 4*t*z**16*(x**2*(2*t**3 + 5*t**2 + t + 1) + y**2*(t*(-2*t**2 + t + 4) + 6)) - t*z**12*(x**6*(t + 1)*(t*(t*(t*(t*(4*t + 25) + 62) + 54) + 6) + 17) + x**4*y**2*(t*(t*(t*(t*(t*(4*t - 9) - 61) + 304) + 392) + 333) + 45) + x**2*y**4*(t*(t*(-t*(t*(t*(4*t + 53) + 375) - 288) + 632) + 361) + 159) + y**6*(-t*(t*(t*(t*(t*(4*t + 31) + 107) + 4) - 252) - 59) + 171)) + 6*x**14*y**4 + 6*x**12*y**6 + 12*x**8*y**10 + 12*x**6*y**12 + 6*x**2*y**16 + 6*y**18 - z**14*(x**4*(t**2*(t + 2)*(t*(4*t*(t + 4) + 17) + 10) + 3) + x**2*y**2*(2*t*(t*(t*(78 - 31*t) + 82) + 12) + 6) + y**4*(t*(t*(-t*(2*t - 1)*(t*(2*t + 13) + 28) + 148) + 24) + 15)) - z**10*(x**8*(t*(t*(t*(t*(t*(t*(13*t + 71) + 146) + 137) + 75) + 49) + 10) + 3) + x**6*y**2*(t*(t*(t*(t*(2*t*(t*(10 - 7*t) + 97) + 647) + 838) + 202) + 138) - 9) + x**4*y**4*(2*t*(t*(t*(-t*(t*(t*(62*t + 139) + 100) - 641) + 814) + 277) + 72) + 18) + x**2*y**6*(t*(t*(t*(-2*t**2*(t*(57*t + 134) + 23) + 359*t + 1222) + 782) + 30) + 51) + y**8*(t*(t*(t*(t*(t*(-t*(17*t + 57) + 2) + 87) + 25) + 449) - 30) + 45)) - z**8*(x**10*(t*(t*(t*(t*(t*(t*(17*t + 79) + 131) + 121) + 103) + 37) + 13) + 3) + x**8*y**2*(t*(t*(t*(t*(t*(t*(t + 55) + 496) + 923) + 549) + 469) + 6) + 21) + x**6*y**4*(t*(t*(t*(t*(t*(-4*t*(50*t + 147) + 1165) + 2246) + 1562) + 708) + 117) + 30) + x**4*y**6*(t*(t*(t*(t*(t*(-4*t*(71*t + 200) + 1299) + 2096) + 1590) + 968) + 183) - 12) + x**2*y**8*(t*(t*(t*(t*(-3*t*(19*t*(t + 3) - 68) + 1109) + 719) + 471) + 290) - 45) + y**10*(t*(t*(t*(t*(11*t**2*(t + 3) + t + 105) + 325) - 157) + 207) - 21)) - z**6*(t*x**12*(t*(t*(3*t*(t*(t*(3*t + 13) + 25) + 32) + 71) + 29) + 17) + 2*t*x**10*y**2*(t*(t*(t*(t*(t*(t + 58) + 246) + 226) + 331) + 110) + 36) + t*x**4*y**8*(t*(t*(t*(t*(t*(543 - 85*t) + 1167) + 1916) + 1105) + 289) + 105) + x**8*y**4*(t*(t*(t*(t*(t*(t*(185 - 207*t) + 1033) + 1990) + 1507) + 391) + 111) + 30) + x**6*y**6*(2*t*(t*(t*(t*(t*(t*(133 - 194*t) + 926) + 1317) + 858) + 243) + 62) + 30) + x**2*y**10*(2*t*(t*(t*(t*(t*(t*(41*t + 189) + 270) + 239) + 255) + 31) - 32) + 30) + y**12*(t*(t*(t*(t*(t*(t*(11*t + 73) + 185) + 74) - 67) + 171) - 141) + 30)) - z**4*(t*x**6*y**8*(t*(t*(t*(t*(t*(139*t + 1287) + 1948) + 1012) + 521) + 181) - 48) + x**14*(t**2*(t + 2)*(t*(t*(t*(t + 4) + 20) + 5) + 17) + 3) + x**12*y**2*(t*(t*(t*(t*(t*(9*t*(t + 6) + 146) + 377) + 203) + 150) + 78) - 9) + x**10*y**4*(t*(t*(t*(t*(t*(t*(41*t + 37) + 874) + 982) + 563) + 467) + 78) - 18) + x**8*y**6*(t*(t*(t*(t*(t*(t*(17*t + 657) + 1468) + 1670) + 835) + 283) + 128) - 18) + x**4*y**10*(t*(t*(t*(t*(t*(t*(179*t + 1189) + 1354) + 562) + 65) - 205) - 138) + 18) + x**2*y**12*(t*(t*(t*(t*(t*(t*(59*t + 326) + 858) + 325) - 271) - 186) - 82) - 21) + y**14*(t*(t*(t*(t*(t*(t*(3*t + 28) + 116) + 171) - 7) - 164) + 24) - 27)) - z**2*(t*x**16*(t**2*(t*(t + 8) + 20) + 7) + 2*t*x**8*y**8*(t*(t*(t*(t*(t*(132*t + 383) + 610) + 96) + 112) - 59) - 14) + t*y**16*(t*(2*t + 7)*(t*(t*(t + 8) + 8) - 6) - 63) + x**14*y**2*(t**2*(t*(t*(2*t*(t + 2)*(t + 4) + 87) + 106) + 50) + 15) + x**12*y**4*(t*(t*(t*(t*(t*(t*(25*t + 91) + 193) + 189) + 391) + 89) + 15) + 15) + x**10*y**6*(t*(t*(t*(t*(2*t*(32*t*(2*t + 7) + 127) + 809) + 296) + 62) - 2) + 21) + x**6*y**10*(t*(t*(t*(t*(2*t*(t*(100*t + 533) + 569) + 527) - 628) - 200) - 66) - 21) + x**4*y**12*(t*(t*(t*(t*(t*(t*(79*t + 437) + 1171) + 307) - 439) - 457) - 75) - 15) + x**2*y**14*(t*(t*(t*(t*(2*t*(t*(3*t + 29) + 184) + 433) - 206) - 280) - 76) - 15))) - (t*(x**2 + y**2) + z**2)*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**3)/(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**(7/2),
            y*z*(mu*(x**18*(6*t**4 + 16*t**3 - 12*t - 6) + x**16*(t*z**2*(t*(2*t + 7)*(t*(t*(t + 8) + 8) - 6) - 63) + y**2*(t*(t*(t*(t*(t*(2*t + 23) + 102) + 46) - 82) - 49) - 6)) + x**14*(t*y**4*(t*(t*(t*(3*t + 16)*(t*(t + 10) + 14) - 77) - 170) - 84) + y**2*z**2*(t*(t*(t*(t*(2*t*(t*(3*t + 29) + 184) + 433) - 206) - 280) - 76) - 15) + z**4*(t*(t*(t*(t*(t*(t*(3*t + 28) + 116) + 171) - 7) - 164) + 24) - 27)) + x**12*(y**6*(t*(t*(t*(t*(t*(t*(31*t + 184) + 406) + 156) - 173) - 204) - 52) - 12) + y**4*z**2*(t*(t*(t*(t*(t*(t*(79*t + 437) + 1171) + 307) - 439) - 457) - 75) - 15) + y**2*z**4*(t*(t*(t*(t*(t*(t*(59*t + 326) + 858) + 325) - 271) - 186) - 82) - 21) + z**6*(t*(t*(t*(t*(t*(t*(11*t + 73) + 185) + 74) - 67) + 171) - 141) + 30)) + x**10*(y**8*(t*(t*(t*(t*(t*(68*t**2 + 286*t + 355) + 84) - 130) - 98) - 49) - 12) + y**6*z**2*(t*(t*(t*(t*(2*t*(t*(100*t + 533) + 569) + 527) - 628) - 200) - 66) - 21) + y**4*z**4*(t*(t*(t*(t*(t*(t*(179*t + 1189) + 1354) + 562) + 65) - 205) - 138) + 18) + y**2*z**6*(2*t*(t*(t*(t*(t*(t*(41*t + 189) + 270) + 239) + 255) + 31) - 32) + 30) + z**8*(t*(t*(t*(t*(11*t**2*(t + 3) + t + 105) + 325) - 157) + 207) - 21)) + x**8*(t*y**10*(t*(t*(t*(t*(8*t*(8*t + 27) + 161) + 96) + 34) - 28) - 39) + 2*t*y**8*z**2*(t*(t*(t*(t*(t*(132*t + 383) + 610) + 96) + 112) - 59) - 14) + t*y**6*z**4*(t*(t*(t*(t*(t*(139*t + 1287) + 1948) + 1012) + 521) + 181) - 48) + t*y**4*z**6*(t*(t*(t*(t*(t*(543 - 85*t) + 1167) + 1916) + 1105) + 289) + 105) + y**2*z**8*(t*(t*(t*(t*(-3*t*(19*t*(t + 3) - 68) + 1109) + 719) + 471) + 290) - 45) + z**10*(t*(t*(t*(t*(t*(-t*(17*t + 57) + 2) + 87) + 25) + 449) - 30) + 45)) + x**6*(t*z**12*(-t*(t*(t*(t*(t*(4*t + 31) + 107) + 4) - 252) - 59) + 171) + y**12*(t*(t*(t*(t*(t*(t*(25*t + 76) + 82) + 54) + 81) + 16) + 8) - 6) + y**10*z**2*(t*(t*(t*(t*(2*t*(32*t*(2*t + 7) + 127) + 809) + 296) + 62) - 2) + 21) + y**8*z**4*(t*(t*(t*(t*(t*(t*(17*t + 657) + 1468) + 1670) + 835) + 283) + 128) - 18) + y**6*z**6*(2*t*(t*(t*(t*(t*(t*(133 - 194*t) + 926) + 1317) + 858) + 243) + 62) + 30) + y**4*z**8*(t*(t*(t*(t*(t*(-4*t*(71*t + 200) + 1299) + 2096) + 1590) + 968) + 183) - 12) + y**2*z**10*(t*(t*(t*(-2*t**2*(t*(57*t + 134) + 23) + 359*t + 1222) + 782) + 30) + 51)) + x**4*(t*y**2*z**12*(t*(t*(-t*(t*(t*(4*t + 53) + 375) - 288) + 632) + 361) + 159) + y**14*(t*(t*(t*(t*(t*(t*(t + 6) + 50) + 22) - 3) + 62) + 12) - 6) + y**12*z**2*(t*(t*(t*(t*(t*(t*(25*t + 91) + 193) + 189) + 391) + 89) + 15) + 15) + y**10*z**4*(t*(t*(t*(t*(t*(t*(41*t + 37) + 874) + 982) + 563) + 467) + 78) - 18) + y**8*z**6*(t*(t*(t*(t*(t*(t*(185 - 207*t) + 1033) + 1990) + 1507) + 391) + 111) + 30) + y**6*z**8*(t*(t*(t*(t*(t*(-4*t*(50*t + 147) + 1165) + 2246) + 1562) + 708) + 117) + 30) + y**4*z**10*(2*t*(t*(t*(-t*(t*(t*(62*t + 139) + 100) - 641) + 814) + 277) + 72) + 18) + z**14*(t*(t*(-t*(2*t - 1)*(t*(2*t + 13) + 28) + 148) + 24) + 15)) + x**2*(t*y**16*(t*(t + 2)*(t*(t + 6) + 2) + 9) + 2*t*y**10*z**6*(t*(t*(t*(t*(t*(t + 58) + 246) + 226) + 331) + 110) + 36) + t*y**4*z**12*(t*(t*(t*(t*(t*(4*t - 9) - 61) + 304) + 392) + 333) + 45) + 4*t*z**16*(t*(-2*t**2 + t + 4) + 6) + y**14*z**2*(t**2*(t*(t*(2*t*(t + 2)*(t + 4) + 87) + 106) + 50) + 15) + y**12*z**4*(t*(t*(t*(t*(t*(9*t*(t + 6) + 146) + 377) + 203) + 150) + 78) - 9) + y**8*z**8*(t*(t*(t*(t*(t*(t*(t + 55) + 496) + 923) + 549) + 469) + 6) + 21) + y**6*z**10*(t*(t*(t*(t*(2*t*(t*(10 - 7*t) + 97) + 647) + 838) + 202) + 138) - 9) + y**2*z**14*(2*t*(t*(t*(78 - 31*t) + 82) + 12) + 6)) + (y**2 + z**2)*(t*y**2 + z**2)*(y**4 + y**2*z**2*(t*(t + 2) - 1) + z**4)*(4*t**2*z**10 + 4*t*y**10 + 8*t*y**6*z**4*(t*(t + 1)**2 + 1) + 4*t*y**2*z**8*(t*(t + 1)**2 + 1) + y**8*z**2*(t**2*(t*(t + 4) + 12) + 3) + y**4*z**6*(t**2*(t*(9*t + 20) + 8) + 3))) + (t*(x**2 + y**2) + z**2)*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**3)/(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**(7/2),
            (x - y)*(x + y)*(mu*(-2*t**7*x**12*y**6 - 6*t**7*x**10*y**8 - 6*t**7*x**8*y**10 - 2*t**7*x**6*y**12 - t**6*x**14*y**4 - 15*t**6*x**12*y**6 - 48*t**6*x**10*y**8 - 48*t**6*x**8*y**10 - 15*t**6*x**6*y**12 - t**6*x**4*y**14 - 16*t**5*x**14*y**4 - 78*t**5*x**12*y**6 - 170*t**5*x**10*y**8 - 170*t**5*x**8*y**10 - 78*t**5*x**6*y**12 - 16*t**5*x**4*y**14 - 5*t**4*x**16*y**2 - 44*t**4*x**14*y**4 - 148*t**4*x**12*y**6 - 187*t**4*x**10*y**8 - 187*t**4*x**8*y**10 - 148*t**4*x**6*y**12 - 44*t**4*x**4*y**14 - 5*t**4*x**2*y**16 - 8*t**3*x**16*y**2 - 16*t**3*x**14*y**4 + 18*t**3*x**12*y**6 + 110*t**3*x**10*y**8 + 110*t**3*x**8*y**10 + 18*t**3*x**6*y**12 - 16*t**3*x**4*y**14 - 8*t**3*x**2*y**16 + 4*t**3*z**18 + 18*t**2*x**16*y**2 + 67*t**2*x**14*y**4 + 285*t**2*x**12*y**6 + 398*t**2*x**10*y**8 + 398*t**2*x**8*y**10 + 285*t**2*x**6*y**12 + 67*t**2*x**4*y**14 + 18*t**2*x**2*y**16 + 4*t**2*z**16*(t + 2)*(2*t**2 + 1)*(x**2 + y**2) + 4*t*x**18 + 16*t*x**16*y**2 + 124*t*x**14*y**4 + 210*t*x**12*y**6 + 326*t*x**10*y**8 + 326*t*x**8*y**10 + 210*t*x**6*y**12 + 124*t*x**4*y**14 + 16*t*x**2*y**16 + 4*t*y**18 + t*z**14*(x**4*(t*(t*(t*(t*(4*t*(t + 4) + 33) + 48) + 32) + 4) + 7) + x**2*y**2*(2*t**2*(t*(t*(2*t + 1)*(2*t + 7) + 90) + 16) + 22) + y**4*(t*(t*(t*(t*(4*t*(t + 4) + 33) + 48) + 32) + 4) + 7)) + t*z**10*(x**8*(t*(t*(t*(t*(t*(17*t + 83) + 147) + 128) + 79) + 33) + 17) + x**6*y**2*(t*(t*(t*(t*(4*t*(5*t + 37) + 627) + 688) + 166) + 368) - 1) + x**4*y**4*(2*t*(t*(t*(t*(t*(33 - 13*t) + 618) + 422) + 149) + 329) - 52) + x**2*y**6*(t*(t*(t*(t*(4*t*(5*t + 37) + 627) + 688) + 166) + 368) - 1) + y**8*(t*(t*(t*(t*(t*(17*t + 83) + 147) + 128) + 79) + 33) + 17)) + t*z**4*(x**2 + y**2)*(x**12*(t*(t*(t*(t*(t + 7) + 36) + 64) + 19) + 17) + x**10*y**2*(4*t*(-t*(t*(t*(t + 12) - 2) - 114) + 104) + 36) + x**8*y**4*(t*(t*(-t*(t*(t*(31*t + 193) + 806) - 820) + 1683) + 517) + 170) - x**6*y**6*(2*t*(t*(t*(t*(t*(29*t + 482) + 63) - 612) - 781) - 554) - 134) + x**4*y**8*(t*(t*(-t*(t*(t*(31*t + 193) + 806) - 820) + 1683) + 517) + 170) + x**2*y**10*(4*t*(-t*(t*(t*(t + 12) - 2) - 114) + 104) + 36) + y**12*(t*(t*(t*(t*(t + 7) + 36) + 64) + 19) + 17)) + 15*x**16*y**2 + 30*x**14*y**4 + 66*x**12*y**6 + 81*x**10*y**8 + 81*x**8*y**10 + 66*x**6*y**12 + 30*x**4*y**14 + 15*x**2*y**16 + z**12*(x**2 + y**2)*(x**4*(t**2*(t*(t*(t*(t*(13*t + 58) + 92) + 77) + 63) + 30) + 3) + x**2*y**2*(2*t*(t*(t*(t*(t*(-t*(t - 40) + 110) + 31) + 151) - 4) + 6) + 6) + y**4*(t**2*(t*(t*(t*(t*(13*t + 58) + 92) + 77) + 63) + 30) + 3)) + z**8*(x**2 + y**2)*(x**8*(t*(t*(t*(t*(t*(t*(9*t + 55) + 139) + 161) + 79) + 45) + 13) + 3) + x**6*y**2*(t*(t*(t*(t*(2*t*(t*(16 - 6*t) + 187) + 755) + 696) + 42) + 138) - 9) + x**4*y**4*(2*t*(t*(t*(t*(t*(-3*t*(35*t + 9) + 323) + 798) + 357) + 77) + 101) - 24) + x**2*y**6*(t*(t*(t*(t*(2*t*(t*(16 - 6*t) + 187) + 755) + 696) + 42) + 138) - 9) + y**8*(t*(t*(t*(t*(t*(t*(9*t + 55) + 139) + 161) + 79) + 45) + 13) + 3)) + z**6*(x**12*(t*(t*(t*(t*(t*(t*(t + 15) + 66) + 113) + 87) + 41) + 10) + 3) + x**10*y**2*(2*t*(t*(t*(t*(t*(t*(2*t - 3) + 20) + 384) + 396) + 189) + 8) + 24) + x**8*y**4*(t*(t*(t*(t*(t*(-t*(97*t + 723) + 340) + 2405) + 2205) + 807) + 28) + 75) + x**6*y**6*(4*t*(t*(t*(t*(t*(-t*(70*t + 327) + 217) + 835) + 726) + 267) + 5) + 108) + x**4*y**8*(t*(t*(t*(t*(t*(-t*(97*t + 723) + 340) + 2405) + 2205) + 807) + 28) + 75) + x**2*y**10*(2*t*(t*(t*(t*(t*(t*(2*t - 3) + 20) + 384) + 396) + 189) + 8) + 24) + y**12*(t*(t*(t*(t*(t*(t*(t + 15) + 66) + 113) + 87) + 41) + 10) + 3)) + z**2*(x**16*(t**2*(t*(t + 8) + 24) + 3) + x**14*y**2*(t*(t*(t*(-t*(9*t + 46) + 66) + 160) + 123) - 6) + x**12*y**4*(t*(-t*(t*(t*(t*(t*(3*t + 17) + 235) + 239) - 427) - 853) + 219) + 3) + x**10*y**6*(t*(-t*(t*(t*(t*(32*t*(t + 6) + 703) + 386) - 1534) - 1360) + 429) + 6) - x**8*y**8*(2*t*(t*(t*(t*(3*t*(t*(7*t + 69) + 167) - 62) - 741) - 931) - 261) + 12) + x**6*y**10*(t*(-t*(t*(t*(t*(32*t*(t + 6) + 703) + 386) - 1534) - 1360) + 429) + 6) + x**4*y**12*(t*(-t*(t*(t*(t*(t*(3*t + 17) + 235) + 239) - 427) - 853) + 219) + 3) + x**2*y**14*(t*(t*(t*(-t*(9*t + 46) + 66) + 160) + 123) - 6) + y**16*(t**2*(t*(t + 8) + 24) + 3))) + (t*z**2 + x**2 + y**2)*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**3)/(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**(7/2)
        ))

    # Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+2, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(phi)
    lset_approx = GridFunction(H1(mesh, order=1, dirichlet=[]))
    InterpolateToP1(phi, lset_approx)
    ci = CutInfo(mesh, lset_approx)

    ba_IF = ci.GetElementsOfType(IF)
    ba_IF_band = BitArray(mesh.ne)
    update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

    # FESpace: Taylor-Hood element
    V = VectorH1(mesh, order=order, dirichlet=[])
    Q = H1(mesh, order=order - 1, dirichlet=[])

    # define projection matrix
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ba_IF, deformation=deformation)
    dX = dx(definedonelements=ba_IF_band, deformation=deformation)
    dX2 = dx(definedonelements=ba_IF, deformation=deformation)

    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)
    h = specialcf.mesh_size
    tau = 1.0 / (h * h)
    rho_u = 1.0 / h
    rho_p = 1.0 * h

    gfu_prevs = [GridFunction(V) for i in range(time_order)]

    out_errs = {'ts': [], 'l2us': [], 'h1us': [], 'l2ps': [], 'h1ps': []}

    if out:
        gfu_out = GridFunction(V)
        gfp_out = GridFunction(Q)
        if fname:
            filename = f"./vtk_out/diffusion/moving-ns-{fname}"
        else:
            filename = "./vtk_out/diffusion/moving-ns"
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, phi, gfu_out, coef_u, gfp_out, coef_p],
                        names=["P1-levelset", "phi", "u", "uSol", "p", "pSol"],
                        filename=filename,
                        subdivision=0)

    for j in range(time_order):
        # fix levelset
        t.Set(-j * dt)
        t_curr = -j * dt

        deformation = lsetmeshadap.CalcDeformation(phi)

        # solve elliptic problem on a fixed surface to get u with normal extension
        InterpolateToP1(phi, lset_approx)
        ci = CutInfo(mesh, lset_approx)

        ba_IF.Clear()
        ba_IF |= ci.GetElementsOfType(IF)
        update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

        VG = Compress(V, GetDofsOfElements(V, ba_IF_band))

        # helper grid functions
        n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=phi, vel_space=V)

        gfu_el = GridFunction(VG)

        u_el, v_el = VG.TnT()

        a_el = BilinearForm(VG, symmetric=True)
        a_el += InnerProduct(Pmat * u_el, Pmat * v_el) * ds
        a_el += 2 * mu * (InnerProduct(Pmat * Sym(grad(u_el)) * Pmat - (u_el * n) * Hmat,
                                       Pmat * Sym(grad(v_el)) * Pmat - (v_el * n) * Hmat)) * ds
        a_el += (tau * ((u_el * n_k) * (v_el * n_k))) * ds
        a_el += (rho_u * InnerProduct(grad(u_el) * n, grad(v_el) * n)) * dX

        f_el = LinearForm(VG)
        f_el += InnerProduct(coef_f_el, Pmat * v_el) * ds

        with TaskManager():
            pre_a_el = Preconditioner(a_el, "bddc")

            a_el.Assemble()
            f_el.Assemble()

            solvers.CG(mat=a_el.mat, rhs=f_el.vec, pre=pre_a_el.mat, sol=gfu_el.vec, printrates=False)
            sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

            gfu_prevs[j].Set(gfu_el)

        with TaskManager():
            l2u, h1u = errors_u(mesh, ds, Pmat, gfu_prevs[j], coef_u)
        append_errors(t_curr, l2u, h1u, 0, 0, **out_errs)

    # TIME MARCHING
    t.Set(0.0)
    t_curr = 0.0

    if out:
        gfu_out.Set(gfu_prevs[0])
        gfp_out.Set(coef_p)
        vtk.Do(time=t_curr)

    i = 1

    while t_curr < tfinal - dt:
        t.Set(t_curr + dt)
        t_curr += dt
        with TaskManager():
            deformation = lsetmeshadap.CalcDeformation(phi)

            InterpolateToP1(phi, lset_approx)
            ci = CutInfo(mesh, lset_approx)

            ba_IF.Clear()
            ba_IF |= ci.GetElementsOfType(IF)
            update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

            VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
            QG = Compress(Q, GetDofsOfElements(Q, ba_IF))
            u, v = VG.TnT()
            p, q = QG.TnT()

            # helper grid functions
            n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=phi, vel_space=V)

        gfu_approx = GridFunction(VG)
        if time_order == 1:
            gfu_approx.Set(Pmat * gfu_prevs[0])
        elif time_order == 2:
            gfu_approx.Set(2 * Pmat * gfu_prevs[0] - Pmat * gfu_prevs[1])
        else:
            gfu_approx.Set(3 * Pmat * gfu_prevs[0] - 3 * Pmat * gfu_prevs[1] + Pmat * gfu_prevs[2])

        a = BilinearForm(VG, symmetric=False)
        a += bdf_coeff[0]/dt * InnerProduct(u, Pmat * v) * ds
        a += wN * InnerProduct(Hmat * u, Pmat * v) * ds
        a += 0.5 * InnerProduct((Pmat * grad(u) * Pmat - (u * n) * Hmat) * gfu_approx, v) * ds
        a += (-0.5) * InnerProduct((Pmat * grad(v) * Pmat - (v * n) * Hmat) * gfu_approx, u) * ds
        a += (-0.5) * InnerProduct(coef_g * u, Pmat * v) * ds
        a += 2.0 * mu * (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                                      Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        a += tau * InnerProduct(n_k, u) * InnerProduct(n_k, v) * ds
        a += rho_u * InnerProduct(grad(u) * n, grad(v) * n) * dX

        # pressure mass-convection-total_stab_tests_diffusion matrix
        ap = BilinearForm(QG, symmetric=False)
        # mass part
        ap += bdf_coeff[0]/dt * p * q * ds
        # total_stab_tests_diffusion
        ap += 2 * mu * InnerProduct(Pmat * grad(p), Pmat * grad(q)) * ds
        # convection
        ap += InnerProduct(Pmat * grad(p), Pmat * gfu_approx) * q * ds
        # normal gradient in the bulk stabilization
        # SHOULD IT BE rho_p OR rho_u?
        ap += rho_p * ((grad(p) * n) * (grad(q) * n)) * dX2

        # pressure total_stab_tests_diffusion matrix
        pd = BilinearForm(QG, symmetric=True)
        # total_stab_tests_diffusion
        pd += InnerProduct(Pmat * grad(p), Pmat * grad(q)) * ds
        # normal gradient in the bulk stabilization
        pd += rho_p * ((grad(p) * n) * (grad(q) * n)) * dX2

        # stabilized pressure mass matrix
        sq = BilinearForm(QG, symmetric=True)
        sq += p * q * ds
        sq += rho_p * ((grad(p) * n) * (grad(q) * n)) * dX2

        b = BilinearForm(trialspace=VG, testspace=QG)
        b += InnerProduct(u, Pmat * grad(q)) * ds

        c = BilinearForm(QG, symmetric=True)
        c += rho_p * (grad(p) * n) * (grad(q) * n) * dX2

        f = LinearForm(VG)
        f += InnerProduct(coef_f, Pmat * v) * ds
        for j in range(time_order):
            f += (-1.0) * bdf_coeff[j+1]/dt * InnerProduct(gfu_prevs[j], Pmat * v) * ds

        g = LinearForm(QG)
        g += (-1.0) * coef_g * q * ds

        with TaskManager():
            pre_a = Preconditioner(a, "bddc")
            pre_pd = Preconditioner(pd, "bddc")
            pre_sq = Preconditioner(sq, "bddc")

            a.Assemble()
            ap.Assemble()
            pd.Assemble()
            b.Assemble()
            c.Assemble()
            pd.Assemble()
            sq.Assemble()

            f.Assemble()
            g.Assemble()

            K = BlockMatrix([[a.mat, b.mat.T],
                             [b.mat, -c.mat]])

            inva = CGSolver(a.mat, pre_a.mat, maxsteps=20, precision=1e-6)
            invpd = CGSolver(pd.mat, pre_pd.mat, maxsteps=10, precision=1e-6)
            invsq = CGSolver(sq.mat, pre_sq.mat, maxsteps=10, precision=1e-6)
            invms = invsq @ ap.mat @ invpd

            C = BlockMatrix([[inva, inva @ b.mat.T @ invms],
                             [None, -invms]])

            rhs = BlockVector([f.vec,
                               g.vec])

            for j in range(time_order-1):
                gfu_prevs[-1 - j].vec.data = gfu_prevs[-2 - j].vec

            gfu = GridFunction(VG)
            gfp = GridFunction(QG)
            sol = BlockVector([gfu.vec,
                               gfp.vec])

            solvers.GMRes(A=K, b=rhs, pre=C, x=sol, printrates=False, maxsteps=100, reltol=1e-12)

            # making numerical pressure mean zero
            renormalize(QG, mesh, ds, gfp)

            if out:
                gfu_out.Set(gfu)
                gfp_out.Set(gfp)
                vtk.Do(time=t_curr)

            l2u, h1u, l2p, h1p = errors(mesh, ds, Pmat, gfu, gfp, coef_u, coef_p)

            gfu_prevs[0].Set(gfu)

        append_errors(t_curr, l2u, h1u, l2p, h1p, **out_errs)

        print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        i += 1

    return h_approx, dt, out_errs['ts'], out_errs['l2us'], out_errs['h1us'], out_errs['l2ps'], out_errs['h1ps']


def moving_ns_for_demos(order, unif_ref, bbox_sz, tfinal, time_order=1,
                        mu=0.5, rho=1.0, fname=None, test_name="advect"):
    bdf_coeffs = [[1, -1],
                  [3 / 2, -2, 1 / 2],
                  [11 / 6, -3, 3 / 2, -1 / 3]]

    bdf_coeff = bdf_coeffs[time_order - 1]

    c_delta = (time_order + 0.5)
    # MESH
    mesh = background_mesh(bbox_sz)

    h_approx = bbox_sz * 2 ** (1 - unif_ref)

    t = Parameter(0.0)

    dt = h_approx ** (order / time_order) / 4

    if test_name == "advect-nonsol-demo":
        vel = 0.2
        w1 = CoefficientFunction(vel)
        w2 = CoefficientFunction(0.0)
        w3 = CoefficientFunction(0.0)
        w = CoefficientFunction((w1, w2, w3))

        R = 1.0
        phi = -R + sqrt((-t*w1 + x)**2 + (-t*w2 + y)**2 + (-t*w3 + z)**2)
        refine_around_lset(mesh, unif_ref, phi, vel, c_delta, tfinal, dt)

        wN = (-t*(w1**2 + w2**2 + w3**2) + w1*x + w2*y + w3*z)/sqrt(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)

        coef_u = CoefficientFunction((
                x * (t * w3 - z) / sqrt(
            t ** 2 * (w1 ** 2 + w2 ** 2 + w3 ** 2) - 2 * t * (w1 * x + w2 * y + w3 * z) + x ** 2 + y ** 2 + z ** 2),
                y * (-t * w3 + z) / sqrt(
                    t ** 2 * (w1 ** 2 + w2 ** 2 + w3 ** 2) - 2 * t * (w1 * x + w2 * y + w3 * z) + x ** 2 + y ** 2 + z ** 2),
                (-t * w1 * x + t * w2 * y + x ** 2 - y ** 2) / sqrt(
                    t ** 2 * (w1 ** 2 + w2 ** 2 + w3 ** 2) - 2 * t * (w1 * x + w2 * y + w3 * z) + x ** 2 + y ** 2 + z ** 2)
        ))
        coef_p = y*(-t*w1 + x) + z

        coef_f = CoefficientFunction((
            (2*mu + rho*(-t*(w1**2 + w2**2 + w3**2) + w1*x + w2*y + w3*z))*(-t*w1*w2*y + t*w2**2*x + t*w3**2*x + w1*y**2 + w1*z**2 - w2*x*y - w3*z*(t*w1 + x))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            (2*mu + rho*(-t*(w1**2 + w2**2 + w3**2) + w1*x + w2*y + w3*z))*(t*(w1**2*y - w1*w2*x + w3*(-w2*z + w3*y)) + w2*(x**2 + z**2) - y*(w1*x + w3*z))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            (2*mu + rho*(-t*(w1**2 + w2**2 + w3**2) + w1*x + w2*y + w3*z))*(-t*w3*(w1*x + w2*y) + t*z*(w1**2 + w2**2) + w3*(x**2 + y**2) - z*(w1*x + w2*y))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2
        ))

        coef_f_el = CoefficientFunction((
            (4*mu*(-t*w1 + x) + x*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            -(4*mu*(-t*w2 + y) + y*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            (4*mu*(t*(w1 - w2) - x + y)*(t*(w1 + w2) - x - y) - (t*w1*x - t*w2*y - x**2 + y**2)*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2)
        ))

        coef_g = CoefficientFunction((2*t*(w1**2 + w2**2 + w3**2) - 2*w1*x - 2*w2*y - 2*w3*z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))
    else:
        vel = 1.0
        R = 1.0
        phi = -R + (2*t*(x**2*y**2 + x**2*z**2 + y**2*z**2) + x**4 + y**4 + z**4)**(1/4)
        refine_around_lset(mesh, unif_ref, phi, vel, c_delta, tfinal, dt, band_type="inner")

        wN = CoefficientFunction(
            (-x**2*(y**2 + z**2) - y**2*z**2)/(2*sqrt(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6))
        )

        coef_u = CoefficientFunction((
            -x*z*(t*(x**2 + y**2) + z**2)/sqrt(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6),
            y*z*(t*(x**2 + y**2) + z**2)/sqrt(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6),
            (x - y)*(x + y)*(t*z**2 + x**2 + y**2)/sqrt(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)
        ))

        coef_p = CoefficientFunction(
            y * (-t / 5 + x) + z
        )

        coef_g = CoefficientFunction(
            (x**2*(y**2 + z**2) + y**2*z**2)*(t**3*x**6*y**2 + t**3*x**2*y**6 + 3*t**2*x**6*y**2 + 4*t**2*x**4*y**4 + 3*t**2*x**2*y**6 + 2*t*x**8 + t*x**6*y**2 + 8*t*x**4*y**4 + t*x**2*y**6 + 2*t*y**8 + 2*t*z**8 + 2*t*z**4*(x**4*(2*t + 4) + x**2*y**2*(-t*(t - 7) + 6) + y**4*(2*t + 4)) + 3*x**6*y**2 + 3*x**2*y**6 + z**6*(t + 3)*(t**2 + 1)*(x**2 + y**2) + z**2*(x**2 + y**2)*(x**4*(t + 3)*(t**2 + 1) - x**2*y**2*(t + 1)*(t*(3*t - 14) + 3) + y**4*(t + 3)*(t**2 + 1)))/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**2)
        )

        coef_f = CoefficientFunction((
            x*(4*mu*(-t**7*x**14*y**8 - 9*t**7*x**12*y**10 + 9*t**7*x**10*y**12 + t**7*x**8*y**14 - t**6*x**16*y**6 - 10*t**6*x**14*y**8 - 26*t**6*x**12*y**10 + 16*t**6*x**10*y**12 + 19*t**6*x**8*y**14 + 2*t**6*x**6*y**16 - 8*t**5*x**16*y**6 - 55*t**5*x**14*y**8 + 16*t**5*x**12*y**10 - 51*t**5*x**10*y**12 + 79*t**5*x**8*y**14 + 18*t**5*x**6*y**16 + t**5*x**4*y**18 - 3*t**4*x**18*y**4 - 21*t**4*x**16*y**6 - 19*t**4*x**14*y**8 - 15*t**4*x**12*y**10 - 13*t**4*x**10*y**12 - 15*t**4*x**8*y**14 + 75*t**4*x**6*y**16 + 11*t**4*x**4*y**18 + 3*t**3*x**18*y**4 - 8*t**3*x**16*y**6 + 109*t**3*x**14*y**8 - 121*t**3*x**12*y**10 + 149*t**3*x**10*y**12 - 169*t**3*x**8*y**14 + 16*t**3*x**6*y**16 + 18*t**3*x**4*y**18 + 3*t**3*x**2*y**20 + 2*t**2*x**20*y**2 + 2*t**2*x**18*y**4 + 78*t**2*x**16*y**6 - 39*t**2*x**14*y**8 + 261*t**2*x**12*y**10 - 241*t**2*x**10*y**12 + 144*t**2*x**8*y**14 - 205*t**2*x**6*y**16 + 3*t**2*x**4*y**18 - 5*t**2*x**2*y**20 + 14*t*x**18*y**4 + 2*t*x**16*y**6 + 111*t*x**14*y**8 - 58*t*x**12*y**10 + 155*t*x**10*y**12 - 167*t*x**8*y**14 + 34*t*x**6*y**16 - 87*t*x**4*y**18 - 2*t*x**2*y**20 - 2*t*y**22 - 2*t*z**22 + 12*x**16*y**6 + 36*x**12*y**10 - 24*x**10*y**12 + 24*x**8*y**14 - 36*x**6*y**16 - 12*x**2*y**20 + z**20*(x**2*(t*(t - 2)*(3*t + 1) - 12) + y**2*(t - 3)*(t*(3*t + 4) + 2)) + z**18*(t*x**4*(t*(t*(t + 2)*(t + 9) + 3) - 87) + t*y**4*(t*(t*(t*(t + 11) + 39) - 82) - 43) + x**2*y**2*(2*t + 2)*(t*(t*(t*(t + 10) + 23) - 59) - 6)) + z**16*(t*y**6*(t*(t*(t*(t*(2*t + 61) + 71) - 164) - 97) - 59) + x**6*(t*(t*(t*(t*(2*t*(t + 9) + 75) + 16) - 205) + 34) - 36) + x**4*y**2*(t*(t*(3*t*(2*t*(t*(t + 5) + 67) - 39) - 541) - 83) - 39) + x**2*y**4*(t*(t*(t*(t*(t*(6*t + 73) + 398) - 306) - 416) - 172) - 15)) + z**14*(x**8*(t*(t*(t*(t*(t*(t*(t + 19) + 79) - 15) - 169) + 144) - 167) + 24) - x**6*y**2*(t*(t*(t*(t*(t*(t*(14*t - 85) - 493) - 8) + 547) + 194) + 260) + 3) + x**4*y**4*(t*(t*(3*t*(t*(t*(t*(59 - 4*t) + 280) + 4) - 270) - 925) - 224) + 6) + x**2*y**6*(t*(t*(t*(t*(t*(t*(22*t + 175) + 528) - 134) - 646) - 701) - 132) - 24) + y**8*(t*(t*(t*(t*(t*(t*(19*t + 64) + 70) - 25) - 259) - 181) + 6) - 18)) + z**12*(t*x**8*y**2*(t*(t*(t*(t*(30*t**2 + 94*t + 105) - 84) + 20) - 270) + 21) + t*y**10*(t*(t*(t*(t*(t*(41*t + 150) + 56) - 289) - 207) - 107) - 64) + x**10*(t - 1)*(t*(t*(t*(t*(t*(9*t + 25) - 26) - 39) + 110) - 131) + 24) + x**6*y**4*(t*(t*(t*(t*(t*(t*(95*t + 261) + 1004) - 571) - 1059) - 183) - 234) + 15) + x**4*y**6*(t*(t*(t*(t*(t*(t*(213*t + 532) + 1009) - 536) - 1909) - 670) - 139) - 12) + x**2*y**8*(t*(t*(t*(t*(t*(t*(180*t + 499) + 225) - 351) - 1225) - 556) - 56) - 60)) - z**10*(t*y**12*(t*(t*(-t*(t*(t*(41*t + 150) + 56) - 289) + 207) + 107) + 64) + x**12*(t*(t*(t*(t*(t*(t*(9*t + 26) - 16) + 15) + 121) - 261) + 58) - 36) + x**10*y**2*(t*(t*(t*(t*(t*(2*t*(25*t + 98) + 99) - 197) - 142) - 328) - 179) - 3) + x**8*y**4*(t*(t*(t*(t*(t*(t*(79*t + 57) + 594) - 494) - 681) - 10) - 316) + 15) + x**6*y**6*(t*(t*(t*(t*(t*(-2*t*(49*t + 384) + 453) + 382) + 297) + 197) - 148) + 21) + x**4*y**8*(t*(t*(t*(-t*(t*(t*(283*t + 1349) + 90) - 1555) + 1286) + 527) + 121) - 3) + x**2*y**10*(2*t*(t*(t*(-2*t**2*(t*(47*t + 201) + 84) + 809*t + 276) + 148) + 208) - 42)) + z**8*(-t*x**14*(t*(t*(t*(t*(t*(t + 10) + 55) + 19) - 109) + 39) - 111) + x**12*y**2*(t*(t*(t*(-t*(t*(t*(82*t + 237) + 215) - 34) + 531) + 558) + 128) + 39) + x**10*y**4*(t*(t*(t*(-t*(t*(t*(269*t + 1201) + 327) - 896) + 1300) + 1369) + 236) + 12) + x**8*y**6*(2*t*(t*(t*(-t*(t*(2*t*(51*t + 371) + 333) - 673) + 898) + 545) + 75) + 72) + x**6*y**8*(2*t - 2)*(t*(t*(t*(t*(t*(53*t + 243) + 33) - 389) - 150) - 73) + 3) + x**4*y**10*(t*(t*(t*(t*(t*(t*(283*t + 1349) + 90) - 1555) - 1286) - 527) - 121) + 3) + x**2*y**12*(t*(t*(t*(t*(t*(t*(180*t + 499) + 225) - 351) - 1225) - 556) - 56) - 60) + y**14*(t*(t*(t*(t*(t*(t*(19*t + 64) + 70) - 25) - 259) - 181) + 6) - 18)) + z**6*(t*y**16*(t*(t*(t*(t*(2*t + 61) + 71) - 164) - 97) - 59) - x**16*(t*(t*(t*(t*(t*(t + 8) + 21) + 8) - 78) - 2) - 12) + x**14*y**2*(t*(t*(t*(-t*(t*(t*(4*t + 61) + 356) - 75) + 409) + 302) + 151) + 12) + x**12*y**4*(t*(t*(t*(-t*(t*(t*(201*t + 521) + 1075) - 431) + 2063) + 845) + 291) + 15) + x**10*y**6*(2*t*(t*(t*(-t*(t*(t*(210*t + 949) + 435) - 790) + 1396) + 773) + 123) + 48) + x**8*y**8*(2*t*(t*(t*(-t*(t*(2*t*(51*t + 371) + 333) - 673) + 898) + 545) + 75) + 72) + x**6*y**10*(t*(t*(t*(t*(t*(98*t**2 + 768*t - 453) - 382) - 297) - 197) + 148) - 21) + x**4*y**12*(t*(t*(t*(t*(t*(t*(213*t + 532) + 1009) - 536) - 1909) - 670) - 139) - 12) + x**2*y**14*(t*(t*(t*(t*(t*(t*(22*t + 175) + 528) - 134) - 646) - 701) - 132) - 24)) + z**4*(t*x**18*(t*(-t*(3*t - 3) + 2) + 14) + 2*t*x**14*y**4*(t*(t*(-t*(t*(3*t*(t + 17) + 301) - 94) + 285) + 344) + 64) + t*y**18*(t*(t*(t*(t + 11) + 39) - 82) - 43) + x**16*y**2*(t*(t*(-3*t*(t*(t*(t + 8) + 40) - 45) + 164) + 40) + 6) + x**12*y**6*(t*(t*(t*(-t*(t*(t*(201*t + 521) + 1075) - 431) + 2063) + 845) + 291) + 15) + x**10*y**8*(t*(t*(t*(-t*(t*(t*(269*t + 1201) + 327) - 896) + 1300) + 1369) + 236) + 12) + x**8*y**10*(t*(t*(t*(-t*(t*(t*(79*t + 57) + 594) - 494) + 681) + 10) + 316) - 15) + x**6*y**12*(t*(t*(t*(t*(t*(t*(95*t + 261) + 1004) - 571) - 1059) - 183) - 234) + 15) + x**4*y**14*(t*(t*(3*t*(t*(t*(t*(59 - 4*t) + 280) + 4) - 270) - 925) - 224) + 6) + x**2*y**16*(t*(t*(t*(t*(t*(6*t + 73) + 398) - 306) - 416) - 172) - 15)) + z**2*(2*t**2*x**20 - 6*t*x**18*y**2*(t + 1)*(t*(t - 2) - 2) + t*x**8*y**12*(t*(t*(t*(t*(30*t**2 + 94*t + 105) - 84) + 20) - 270) + 21) + x**16*y**4*(t*(t*(-3*t*(t*(t*(t + 8) + 40) - 45) + 164) + 40) + 6) + x**14*y**6*(t*(t*(t*(-t*(t*(t*(4*t + 61) + 356) - 75) + 409) + 302) + 151) + 12) + x**12*y**8*(t*(t*(t*(-t*(t*(t*(82*t + 237) + 215) - 34) + 531) + 558) + 128) + 39) + x**10*y**10*(t*(t*(t*(-t*(t*(2*t*(25*t + 98) + 99) - 197) + 142) + 328) + 179) + 3) - x**6*y**14*(t*(t*(t*(t*(t*(t*(14*t - 85) - 493) - 8) + 547) + 194) + 260) + 3) + x**4*y**16*(t*(t*(3*t*(2*t*(t*(t + 5) + 67) - 39) - 541) - 83) - 39) + x**2*y**18*(2*t + 2)*(t*(t*(t*(t + 10) + 23) - 59) - 6) + y**20*(t - 3)*(t*(3*t + 4) + 2))) - rho*(x**2*(y**2 + z**2) + y**2*z**2)*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)*(2*t*x**12*y**2 + t*x**8*y**6*(t*(t + 1)**2 + 6) - t*x**2*y**12*(t + 7) + x**10*y**4*(t**2*(t + 5) + 2) - x**6*y**8*(t - 1)*(t**2*(t + 2) + 1) - x**4*y**10*(t**2*(2*t + 7) + 1) - 2*y**14 - 2*z**14 - z**12*(t*(t + 7)*(x**2 + y**2) + 2*y**2) - z**10*(t*y**4*(t*(2*t + 5) + 15) + x**4*(t**2*(2*t + 7) + 1) + x**2*y**2*(t*(t*(4*t + 19) + 4) + 1)) - z**8*(2*t*x**4*y**2*(3*t**3 + 2*t**2 + t - 1) + x**6*(t - 1)*(t**2*(t + 2) + 1) + x**2*y**4*(t*(t*(t*(3*t + 7) + 28) + 4) - 2) + y**6*(2*t*(t*(-t*(t - 2) + 10) + 2) + 4)) + z**6*(t*x**8*(t*(t + 1)**2 + 6) + 2*t*x**2*y**6*(t*(t*(5*t - 20) - 3) - 2) + x**6*y**2*(t*(t*(-t*(t - 11) + 22) + 7) + 1) + x**4*y**4*(t*(-t*(5*t - 20) + 3) + 2) + y**8*(2*t*(t*(t*(t - 2) - 10) - 2) - 4)) + z**4*(2*t*x**6*y**4*(t*(23*t + 5) + 12) - t*y**10*(t*(2*t + 5) + 15) + x**10*(t**2*(t + 5) + 2) + x**8*y**2*(t*(t*(t*(3*t + 11) + 26) + 8) + 2) + x**4*y**6*(t*(-t*(5*t - 20) + 3) + 2) - x**2*y**8*(t*(t*(t*(3*t + 7) + 28) + 4) - 2)) + z**2*(2*t*x**12 + 2*t*x**10*y**2*(t + 1)*(t + 4) - 2*t*x**4*y**8*(3*t**3 + 2*t**2 + t - 1) + x**8*y**4*(t*(t*(t*(3*t + 11) + 26) + 8) + 2) + x**6*y**6*(t*(t*(-t*(t - 11) + 22) + 7) + 1) - x**2*y**10*(t*(t*(4*t + 19) + 4) + 1) - y**12*(t*(t + 7) + 2))))/(4*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**4),
            y*(4*mu*(t**7*x**14*y**8 + 9*t**7*x**12*y**10 - 9*t**7*x**10*y**12 - t**7*x**8*y**14 + 2*t**6*x**16*y**6 + 19*t**6*x**14*y**8 + 16*t**6*x**12*y**10 - 26*t**6*x**10*y**12 - 10*t**6*x**8*y**14 - t**6*x**6*y**16 + t**5*x**18*y**4 + 18*t**5*x**16*y**6 + 79*t**5*x**14*y**8 - 51*t**5*x**12*y**10 + 16*t**5*x**10*y**12 - 55*t**5*x**8*y**14 - 8*t**5*x**6*y**16 + 11*t**4*x**18*y**4 + 75*t**4*x**16*y**6 - 15*t**4*x**14*y**8 - 13*t**4*x**12*y**10 - 15*t**4*x**10*y**12 - 19*t**4*x**8*y**14 - 21*t**4*x**6*y**16 - 3*t**4*x**4*y**18 + 3*t**3*x**20*y**2 + 18*t**3*x**18*y**4 + 16*t**3*x**16*y**6 - 169*t**3*x**14*y**8 + 149*t**3*x**12*y**10 - 121*t**3*x**10*y**12 + 109*t**3*x**8*y**14 - 8*t**3*x**6*y**16 + 3*t**3*x**4*y**18 - 5*t**2*x**20*y**2 + 3*t**2*x**18*y**4 - 205*t**2*x**16*y**6 + 144*t**2*x**14*y**8 - 241*t**2*x**12*y**10 + 261*t**2*x**10*y**12 - 39*t**2*x**8*y**14 + 78*t**2*x**6*y**16 + 2*t**2*x**4*y**18 + 2*t**2*x**2*y**20 - 2*t*x**22 - 2*t*x**20*y**2 - 87*t*x**18*y**4 + 34*t*x**16*y**6 - 167*t*x**14*y**8 + 155*t*x**12*y**10 - 58*t*x**10*y**12 + 111*t*x**8*y**14 + 2*t*x**6*y**16 + 14*t*x**4*y**18 - 2*t*z**22 - 12*x**20*y**2 - 36*x**16*y**6 + 24*x**14*y**8 - 24*x**12*y**10 + 36*x**10*y**12 + 12*x**6*y**16 + z**20*(x**2*(t - 3)*(t*(3*t + 4) + 2) + y**2*(t*(t - 2)*(3*t + 1) - 12)) + z**18*(t*x**4*(t*(t*(t*(t + 11) + 39) - 82) - 43) + t*y**4*(t*(t*(t + 2)*(t + 9) + 3) - 87) + x**2*y**2*(2*t + 2)*(t*(t*(t*(t + 10) + 23) - 59) - 6)) + z**16*(t*x**6*(t*(t*(t*(t*(2*t + 61) + 71) - 164) - 97) - 59) + x**4*y**2*(t*(t*(t*(t*(t*(6*t + 73) + 398) - 306) - 416) - 172) - 15) + x**2*y**4*(t*(t*(3*t*(2*t*(t*(t + 5) + 67) - 39) - 541) - 83) - 39) + y**6*(t*(t*(t*(t*(2*t*(t + 9) + 75) + 16) - 205) + 34) - 36)) + z**14*(x**8*(t*(t*(t*(t*(t*(t*(19*t + 64) + 70) - 25) - 259) - 181) + 6) - 18) + x**6*y**2*(t*(t*(t*(t*(t*(t*(22*t + 175) + 528) - 134) - 646) - 701) - 132) - 24) + x**4*y**4*(t*(t*(3*t*(t*(t*(t*(59 - 4*t) + 280) + 4) - 270) - 925) - 224) + 6) - x**2*y**6*(t*(t*(t*(t*(t*(t*(14*t - 85) - 493) - 8) + 547) + 194) + 260) + 3) + y**8*(t*(t*(t*(t*(t*(t*(t + 19) + 79) - 15) - 169) + 144) - 167) + 24)) + z**12*(t*x**10*(t*(t*(t*(t*(t*(41*t + 150) + 56) - 289) - 207) - 107) - 64) + t*x**2*y**8*(t*(t*(t*(t*(30*t**2 + 94*t + 105) - 84) + 20) - 270) + 21) + x**8*y**2*(t*(t*(t*(t*(t*(t*(180*t + 499) + 225) - 351) - 1225) - 556) - 56) - 60) + x**6*y**4*(t*(t*(t*(t*(t*(t*(213*t + 532) + 1009) - 536) - 1909) - 670) - 139) - 12) + x**4*y**6*(t*(t*(t*(t*(t*(t*(95*t + 261) + 1004) - 571) - 1059) - 183) - 234) + 15) + y**10*(t - 1)*(t*(t*(t*(t*(t*(9*t + 25) - 26) - 39) + 110) - 131) + 24)) + z**10*(t*x**12*(t*(t*(t*(t*(t*(41*t + 150) + 56) - 289) - 207) - 107) - 64) + x**10*y**2*(2*t*(t*(t*(t*(2*t*(t*(47*t + 201) + 84) - 809) - 276) - 148) - 208) + 42) + x**8*y**4*(t*(t*(t*(t*(t*(t*(283*t + 1349) + 90) - 1555) - 1286) - 527) - 121) + 3) + x**6*y**6*(t*(t*(t*(t*(t*(98*t**2 + 768*t - 453) - 382) - 297) - 197) + 148) - 21) + x**4*y**8*(t*(t*(t*(-t*(t*(t*(79*t + 57) + 594) - 494) + 681) + 10) + 316) - 15) + x**2*y**10*(t*(t*(t*(-t*(t*(2*t*(25*t + 98) + 99) - 197) + 142) + 328) + 179) + 3) - y**12*(t*(t*(t*(t*(t*(t*(9*t + 26) - 16) + 15) + 121) - 261) + 58) - 36)) + z**8*(-t*y**14*(t*(t*(t*(t*(t*(t + 10) + 55) + 19) - 109) + 39) - 111) + x**14*(t*(t*(t*(t*(t*(t*(19*t + 64) + 70) - 25) - 259) - 181) + 6) - 18) + x**12*y**2*(t*(t*(t*(t*(t*(t*(180*t + 499) + 225) - 351) - 1225) - 556) - 56) - 60) + x**10*y**4*(t*(t*(t*(t*(t*(t*(283*t + 1349) + 90) - 1555) - 1286) - 527) - 121) + 3) + x**8*y**6*(2*t - 2)*(t*(t*(t*(t*(t*(53*t + 243) + 33) - 389) - 150) - 73) + 3) + x**6*y**8*(2*t*(t*(t*(-t*(t*(2*t*(51*t + 371) + 333) - 673) + 898) + 545) + 75) + 72) + x**4*y**10*(t*(t*(t*(-t*(t*(t*(269*t + 1201) + 327) - 896) + 1300) + 1369) + 236) + 12) + x**2*y**12*(t*(t*(t*(-t*(t*(t*(82*t + 237) + 215) - 34) + 531) + 558) + 128) + 39)) + z**6*(t*x**16*(t*(t*(t*(t*(2*t + 61) + 71) - 164) - 97) - 59) + x**14*y**2*(t*(t*(t*(t*(t*(t*(22*t + 175) + 528) - 134) - 646) - 701) - 132) - 24) + x**12*y**4*(t*(t*(t*(t*(t*(t*(213*t + 532) + 1009) - 536) - 1909) - 670) - 139) - 12) + x**10*y**6*(t*(t*(t*(t*(t*(98*t**2 + 768*t - 453) - 382) - 297) - 197) + 148) - 21) + x**8*y**8*(2*t*(t*(t*(-t*(t*(2*t*(51*t + 371) + 333) - 673) + 898) + 545) + 75) + 72) + x**6*y**10*(2*t*(t*(t*(-t*(t*(t*(210*t + 949) + 435) - 790) + 1396) + 773) + 123) + 48) + x**4*y**12*(t*(t*(t*(-t*(t*(t*(201*t + 521) + 1075) - 431) + 2063) + 845) + 291) + 15) + x**2*y**14*(t*(t*(t*(-t*(t*(t*(4*t + 61) + 356) - 75) + 409) + 302) + 151) + 12) - y**16*(t*(t*(t*(t*(t*(t + 8) + 21) + 8) - 78) - 2) - 12)) + z**4*(t*x**18*(t*(t*(t*(t + 11) + 39) - 82) - 43) + 2*t*x**4*y**14*(t*(t*(-t*(t*(3*t*(t + 17) + 301) - 94) + 285) + 344) + 64) + t*y**18*(t*(-t*(3*t - 3) + 2) + 14) + x**16*y**2*(t*(t*(t*(t*(t*(6*t + 73) + 398) - 306) - 416) - 172) - 15) + x**14*y**4*(t*(t*(3*t*(t*(t*(t*(59 - 4*t) + 280) + 4) - 270) - 925) - 224) + 6) + x**12*y**6*(t*(t*(t*(t*(t*(t*(95*t + 261) + 1004) - 571) - 1059) - 183) - 234) + 15) + x**10*y**8*(t*(t*(t*(-t*(t*(t*(79*t + 57) + 594) - 494) + 681) + 10) + 316) - 15) + x**8*y**10*(t*(t*(t*(-t*(t*(t*(269*t + 1201) + 327) - 896) + 1300) + 1369) + 236) + 12) + x**6*y**12*(t*(t*(t*(-t*(t*(t*(201*t + 521) + 1075) - 431) + 2063) + 845) + 291) + 15) + x**2*y**16*(t*(t*(-3*t*(t*(t*(t + 8) + 40) - 45) + 164) + 40) + 6)) + z**2*(2*t**2*y**20 + t*x**12*y**8*(t*(t*(t*(t*(30*t**2 + 94*t + 105) - 84) + 20) - 270) + 21) + 6*t*x**2*y**18*(t*(-t**2 + t + 4) + 2) + x**20*(t - 3)*(t*(3*t + 4) + 2) + x**18*y**2*(2*t + 2)*(t*(t*(t*(t + 10) + 23) - 59) - 6) + x**16*y**4*(t*(t*(3*t*(2*t*(t*(t + 5) + 67) - 39) - 541) - 83) - 39) - x**14*y**6*(t*(t*(t*(t*(t*(t*(14*t - 85) - 493) - 8) + 547) + 194) + 260) + 3) + x**10*y**10*(t*(t*(t*(-t*(t*(2*t*(25*t + 98) + 99) - 197) + 142) + 328) + 179) + 3) + x**8*y**12*(t*(t*(t*(-t*(t*(t*(82*t + 237) + 215) - 34) + 531) + 558) + 128) + 39) + x**6*y**14*(t*(t*(t*(-t*(t*(t*(4*t + 61) + 356) - 75) + 409) + 302) + 151) + 12) + x**4*y**16*(t*(t*(-3*t*(t*(t*(t + 8) + 40) - 45) + 164) + 40) + 6))) + rho*(x**2*(y**2 + z**2) + y**2*z**2)*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)*(-2*t*y**12*z**2 - t*y**8*z**6*(t*(t + 1)**2 + 6) + t*y**2*z**12*(t + 7) + 2*x**14 + x**12*(t*(t + 7)*(y**2 + z**2) + 2*z**2) + x**10*(t*z**4*(t*(2*t + 5) + 15) + y**4*(t**2*(2*t + 7) + 1) + y**2*z**2*(t*(t*(4*t + 19) + 4) + 1)) + x**8*(2*t*y**4*z**2*(3*t**3 + 2*t**2 + t - 1) + y**6*(t - 1)*(t**2*(t + 2) + 1) + y**2*z**4*(t*(t*(t*(3*t + 7) + 28) + 4) - 2) + z**6*(2*t*(t*(-t*(t - 2) + 10) + 2) + 4)) + x**6*(-t*y**8*(t*(t + 1)**2 + 6) + 2*t*y**2*z**6*(t*(-t*(5*t - 20) + 3) + 2) + y**6*z**2*(t*(t*(t*(t - 11) - 22) - 7) - 1) + y**4*z**4*(t*(t*(5*t - 20) - 3) - 2) + z**8*(2*t*(t*(-t*(t - 2) + 10) + 2) + 4)) + x**4*(-2*t*y**6*z**4*(t*(23*t + 5) + 12) + t*z**10*(t*(2*t + 5) + 15) - y**10*(t**2*(t + 5) + 2) - y**8*z**2*(t*(t*(t*(3*t + 11) + 26) + 8) + 2) + y**4*z**6*(t*(t*(5*t - 20) - 3) - 2) + y**2*z**8*(t*(t*(t*(3*t + 7) + 28) + 4) - 2)) + x**2*(-2*t*y**12 - 2*t*y**10*z**2*(t + 1)*(t + 4) + 2*t*y**4*z**8*(3*t**3 + 2*t**2 + t - 1) - y**8*z**4*(t*(t*(t*(3*t + 11) + 26) + 8) + 2) + y**6*z**6*(t*(t*(t*(t - 11) - 22) - 7) - 1) + y**2*z**10*(t*(t*(4*t + 19) + 4) + 1) + z**12*(t*(t + 7) + 2)) - y**10*z**4*(t**2*(t + 5) + 2) + y**6*z**8*(t - 1)*(t**2*(t + 2) + 1) + y**4*z**10*(t**2*(2*t + 7) + 1) + 2*z**14))/(4*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**4),
            z*(4*mu*(19*t**7*x**14*y**8 + 41*t**7*x**12*y**10 + 41*t**7*x**10*y**12 + 19*t**7*x**8*y**14 + 2*t**6*x**16*y**6 + 64*t**6*x**14*y**8 + 150*t**6*x**12*y**10 + 150*t**6*x**10*y**12 + 64*t**6*x**8*y**14 + 2*t**6*x**6*y**16 + t**5*x**18*y**4 + 61*t**5*x**16*y**6 + 70*t**5*x**14*y**8 + 56*t**5*x**12*y**10 + 56*t**5*x**10*y**12 + 70*t**5*x**8*y**14 + 61*t**5*x**6*y**16 + t**5*x**4*y**18 + 11*t**4*x**18*y**4 + 71*t**4*x**16*y**6 - 25*t**4*x**14*y**8 - 289*t**4*x**12*y**10 - 289*t**4*x**10*y**12 - 25*t**4*x**8*y**14 + 71*t**4*x**6*y**16 + 11*t**4*x**4*y**18 + 3*t**3*x**20*y**2 + 39*t**3*x**18*y**4 - 164*t**3*x**16*y**6 - 259*t**3*x**14*y**8 - 207*t**3*x**12*y**10 - 207*t**3*x**10*y**12 - 259*t**3*x**8*y**14 - 164*t**3*x**6*y**16 + 39*t**3*x**4*y**18 + 3*t**3*x**2*y**20 - 5*t**2*x**20*y**2 - 82*t**2*x**18*y**4 - 97*t**2*x**16*y**6 - 181*t**2*x**14*y**8 - 107*t**2*x**12*y**10 - 107*t**2*x**10*y**12 - 181*t**2*x**8*y**14 - 97*t**2*x**6*y**16 - 82*t**2*x**4*y**18 - 5*t**2*x**2*y**20 + 2*t**2*z**20*(x**2 + y**2) - 2*t*x**22 - 10*t*x**20*y**2 - 43*t*x**18*y**4 - 59*t*x**16*y**6 + 6*t*x**14*y**8 - 64*t*x**12*y**10 - 64*t*x**10*y**12 + 6*t*x**8*y**14 - 59*t*x**6*y**16 - 43*t*x**4*y**18 - 10*t*x**2*y**20 - 2*t*y**22 + t*z**18*(x**4*(t*(-t*(3*t - 3) + 2) + 14) + x**2*y**2*(6*t*(-t**2 + t + 4) + 12) + y**4*(t*(-t*(3*t - 3) + 2) + 14)) - 6*x**20*y**2 - 18*x**14*y**8 - 18*x**8*y**14 - 6*x**2*y**20 - z**16*(x**2 + y**2)*(x**4*(t*(t*(t*(t*(t*(t + 8) + 21) + 8) - 78) - 2) - 12) + x**2*y**2*(t*(t*(t*(t*(2*t*(t + 8) + 99) - 143) - 86) - 38) + 6) + y**4*(t*(t*(t*(t*(t*(t + 8) + 21) + 8) - 78) - 2) - 12)) - z**14*(t*x**8*(t*(t*(t*(t*(t*(t + 10) + 55) + 19) - 109) + 39) - 111) + 2*t*x**4*y**4*(t*(t*(t*(t*(3*t*(t + 17) + 301) - 94) - 285) - 344) - 64) + t*y**8*(t*(t*(t*(t*(t*(t + 10) + 55) + 19) - 109) + 39) - 111) + x**6*y**2*(t*(t*(t*(t*(t*(t*(4*t + 61) + 356) - 75) - 409) - 302) - 151) - 12) + x**2*y**6*(t*(t*(t*(t*(t*(t*(4*t + 61) + 356) - 75) - 409) - 302) - 151) - 12)) - z**12*(x**2 + y**2)*(x**8*(t*(t*(t*(t*(t*(t*(9*t + 26) - 16) + 15) + 121) - 261) + 58) - 36) + x**6*y**2*(t*(t*(t*(t*(t*(t*(73*t + 211) + 231) - 49) - 652) - 297) - 186) - 3) + x**4*y**4*(t*(t*(t*(2*t*(t*(t*(64*t + 155) + 422) - 191) - 1411) - 548) - 105) - 12) + x**2*y**6*(t*(t*(t*(t*(t*(t*(73*t + 211) + 231) - 49) - 652) - 297) - 186) - 3) + y**8*(t*(t*(t*(t*(t*(t*(9*t + 26) - 16) + 15) + 121) - 261) + 58) - 36)) + z**10*(x**12*(t - 1)*(t*(t*(t*(t*(t*(9*t + 25) - 26) - 39) + 110) - 131) + 24) + x**10*y**2*(t*(t*(t*(-t*(t*(2*t*(25*t + 98) + 99) - 197) + 142) + 328) + 179) + 3) + x**8*y**4*(t*(t*(t*(-t*(t*(t*(269*t + 1201) + 327) - 896) + 1300) + 1369) + 236) + 12) + x**6*y**6*(2*t*(t*(t*(-t*(t*(t*(210*t + 949) + 435) - 790) + 1396) + 773) + 123) + 48) + x**4*y**8*(t*(t*(t*(-t*(t*(t*(269*t + 1201) + 327) - 896) + 1300) + 1369) + 236) + 12) + x**2*y**10*(t*(t*(t*(-t*(t*(2*t*(25*t + 98) + 99) - 197) + 142) + 328) + 179) + 3) + y**12*(t - 1)*(t*(t*(t*(t*(t*(9*t + 25) - 26) - 39) + 110) - 131) + 24)) + z**8*(x**2 + y**2)*(x**12*(t*(t*(t*(t*(t*(t*(t + 19) + 79) - 15) - 169) + 144) - 167) + 24) + x**10*y**2*(t - 1)*(t*(t*(t*(t*(t*(29*t + 104) + 130) + 61) + 250) - 164) + 24) + x**8*y**4*(t*(t*(t*(-4*t**2*(3*t*(9*t + 11) + 155) + 563*t + 492) + 424) + 128) + 9) + x**6*y**6*(t*(t*(t*(-2*t**2*(48*t**2 + 676*t + 23) + 783*t + 1304) + 666) + 22) + 63) + x**4*y**8*(t*(t*(t*(-4*t**2*(3*t*(9*t + 11) + 155) + 563*t + 492) + 424) + 128) + 9) + x**2*y**10*(t - 1)*(t*(t*(t*(t*(t*(29*t + 104) + 130) + 61) + 250) - 164) + 24) + y**12*(t*(t*(t*(t*(t*(t*(t + 19) + 79) - 15) - 169) + 144) - 167) + 24)) + z**6*(x**16*(t*(t*(t*(t*(2*t*(t + 9) + 75) + 16) - 205) + 34) - 36) - x**14*y**2*(t*(t*(t*(t*(t*(t*(14*t - 85) - 493) - 8) + 547) + 194) + 260) + 3) + x**12*y**4*(t*(t*(t*(t*(t*(t*(95*t + 261) + 1004) - 571) - 1059) - 183) - 234) + 15) + x**10*y**6*(t*(t*(t*(t*(t*(98*t**2 + 768*t - 453) - 382) - 297) - 197) + 148) - 21) + x**8*y**8*(2*t - 2)*(t*(t*(t*(t*(t*(53*t + 243) + 33) - 389) - 150) - 73) + 3) + x**6*y**10*(t*(t*(t*(t*(t*(98*t**2 + 768*t - 453) - 382) - 297) - 197) + 148) - 21) + x**4*y**12*(t*(t*(t*(t*(t*(t*(95*t + 261) + 1004) - 571) - 1059) - 183) - 234) + 15) - x**2*y**14*(t*(t*(t*(t*(t*(t*(14*t - 85) - 493) - 8) + 547) + 194) + 260) + 3) + y**16*(t*(t*(t*(t*(2*t*(t + 9) + 75) + 16) - 205) + 34) - 36)) + z**4*(x**2 + y**2)*(t*x**16*(t*(t*(t + 2)*(t + 9) + 3) - 87) + t*y**16*(t*(t*(t + 2)*(t + 9) + 3) - 87) + x**14*y**2*(t*(t*(t*(t*(t*(6*t + 29) + 391) - 135) - 544) + 4) - 39) - x**12*y**4*(t*(t*(t*(t*(t*(3*t*(4*t - 57) - 811) + 379) + 675) + 381) + 228) - 45) + x**10*y**6*(t*(t*(t*(t*(t*(t*(225*t + 361) + 198) - 157) - 1234) - 289) + 89) - 57) + x**8*y**8*(2*t*(t*(t*(t*(t*(t*(29*t + 494) - 54) - 699) - 26) - 119) - 105) + 60) + x**6*y**10*(t*(t*(t*(t*(t*(t*(225*t + 361) + 198) - 157) - 1234) - 289) + 89) - 57) - x**4*y**12*(t*(t*(t*(t*(t*(3*t*(4*t - 57) - 811) + 379) + 675) + 381) + 228) - 45) + x**2*y**14*(t*(t*(t*(t*(t*(6*t + 29) + 391) - 135) - 544) + 4) - 39)) + z**2*(x**20*(t*(t - 2)*(3*t + 1) - 12) + x**18*y**2*(2*t + 2)*(t*(t*(t*(t + 10) + 23) - 59) - 6) + x**16*y**4*(t*(t*(t*(t*(t*(6*t + 73) + 398) - 306) - 416) - 172) - 15) + x**14*y**6*(t*(t*(t*(t*(t*(t*(22*t + 175) + 528) - 134) - 646) - 701) - 132) - 24) + x**12*y**8*(t*(t*(t*(t*(t*(t*(180*t + 499) + 225) - 351) - 1225) - 556) - 56) - 60) + x**10*y**10*(2*t*(t*(t*(t*(2*t*(t*(47*t + 201) + 84) - 809) - 276) - 148) - 208) + 42) + x**8*y**12*(t*(t*(t*(t*(t*(t*(180*t + 499) + 225) - 351) - 1225) - 556) - 56) - 60) + x**6*y**14*(t*(t*(t*(t*(t*(t*(22*t + 175) + 528) - 134) - 646) - 701) - 132) - 24) + x**4*y**16*(t*(t*(t*(t*(t*(6*t + 73) + 398) - 306) - 416) - 172) - 15) + x**2*y**18*(2*t + 2)*(t*(t*(t*(t + 10) + 23) - 59) - 6) + y**20*(t*(t - 2)*(3*t + 1) - 12))) + rho*(x**2*(y**2 + z**2) + y**2*z**2)*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)*(t*y**12*z**2*(t + 7) - t*y**6*z**8*(t*(t + 1)**2 + 6) - 2*t*y**2*z**12 + 2*x**14 + x**12*(t*z**2*(t + 7) + y**2*(t*(t + 7) + 2)) + x**10*(t*y**4*(t*(2*t + 5) + 15) + y**2*z**2*(t*(t*(4*t + 19) + 4) + 1) + z**4*(t**2*(2*t + 7) + 1)) + x**8*(2*t*y**2*z**4*(3*t**3 + 2*t**2 + t - 1) + y**6*(2*t*(t*(-t*(t - 2) + 10) + 2) + 4) + y**4*z**2*(t*(t*(t*(3*t + 7) + 28) + 4) - 2) + z**6*(t - 1)*(t**2*(t + 2) + 1)) + x**6*(2*t*y**6*z**2*(t*(-t*(5*t - 20) + 3) + 2) - t*z**8*(t*(t + 1)**2 + 6) + y**8*(2*t*(t*(-t*(t - 2) + 10) + 2) + 4) + y**4*z**4*(t*(t*(5*t - 20) - 3) - 2) + y**2*z**6*(t*(t*(t*(t - 11) - 22) - 7) - 1)) + x**4*(t*y**10*(t*(2*t + 5) + 15) - 2*t*y**4*z**6*(t*(23*t + 5) + 12) + y**8*z**2*(t*(t*(t*(3*t + 7) + 28) + 4) - 2) + y**6*z**4*(t*(t*(5*t - 20) - 3) - 2) - y**2*z**8*(t*(t*(t*(3*t + 11) + 26) + 8) + 2) - z**10*(t**2*(t + 5) + 2)) + x**2*(2*t*y**8*z**4*(3*t**3 + 2*t**2 + t - 1) - 2*t*y**2*z**10*(t + 1)*(t + 4) - 2*t*z**12 + y**12*(t*(t + 7) + 2) + y**10*z**2*(t*(t*(4*t + 19) + 4) + 1) + y**6*z**6*(t*(t*(t*(t - 11) - 22) - 7) - 1) - y**4*z**8*(t*(t*(t*(3*t + 11) + 26) + 8) + 2)) + 2*y**14 + y**10*z**4*(t**2*(2*t + 7) + 1) + y**8*z**6*(t - 1)*(t**2*(t + 2) + 1) - y**4*z**10*(t**2*(t + 5) + 2)))/(4*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**4)
        ))

        coef_f_el = CoefficientFunction((
            x*z*(mu*(-t**7*x**14*y**4 - 25*t**7*x**12*y**6 - 64*t**7*x**10*y**8 - 68*t**7*x**8*y**10 - 31*t**7*x**6*y**12 - 3*t**7*x**4*y**14 - 6*t**6*x**14*y**4 - 76*t**6*x**12*y**6 - 216*t**6*x**10*y**8 - 286*t**6*x**8*y**10 - 184*t**6*x**6*y**12 - 46*t**6*x**4*y**14 - 2*t**6*x**2*y**16 - t**5*x**16*y**2 - 50*t**5*x**14*y**4 - 82*t**5*x**12*y**6 - 161*t**5*x**10*y**8 - 355*t**5*x**8*y**10 - 406*t**5*x**6*y**12 - 202*t**5*x**4*y**14 - 23*t**5*x**2*y**16 - 8*t**4*x**16*y**2 - 22*t**4*x**14*y**4 - 54*t**4*x**12*y**6 - 96*t**4*x**10*y**8 - 84*t**4*x**8*y**10 - 156*t**4*x**6*y**12 - 224*t**4*x**4*y**14 - 102*t**4*x**2*y**16 - 6*t**4*y**18 - 14*t**3*x**16*y**2 + 3*t**3*x**14*y**4 - 81*t**3*x**12*y**6 - 34*t**3*x**10*y**8 + 130*t**3*x**8*y**10 + 173*t**3*x**6*y**12 + 77*t**3*x**4*y**14 - 46*t**3*x**2*y**16 - 16*t**3*y**18 - 4*t**2*x**18 - 4*t**2*x**16*y**2 - 62*t**2*x**14*y**4 - 16*t**2*x**12*y**6 + 28*t**2*x**10*y**8 + 98*t**2*x**8*y**10 + 204*t**2*x**6*y**12 + 170*t**2*x**4*y**14 + 82*t**2*x**2*y**16 - 4*t**2*z**18 - 9*t*x**16*y**2 - 12*t*x**14*y**4 - 8*t*x**12*y**6 + 39*t*x**10*y**8 + 49*t*x**8*y**10 + 52*t*x**6*y**12 + 84*t*x**4*y**14 + 49*t*x**2*y**16 + 12*t*y**18 - 4*t*z**16*(x**2*(2*t**3 + 5*t**2 + t + 1) + y**2*(t*(-2*t**2 + t + 4) + 6)) - t*z**12*(x**6*(t + 1)*(t*(t*(t*(t*(4*t + 25) + 62) + 54) + 6) + 17) + x**4*y**2*(t*(t*(t*(t*(t*(4*t - 9) - 61) + 304) + 392) + 333) + 45) + x**2*y**4*(t*(t*(-t*(t*(t*(4*t + 53) + 375) - 288) + 632) + 361) + 159) + y**6*(-t*(t*(t*(t*(t*(4*t + 31) + 107) + 4) - 252) - 59) + 171)) + 6*x**14*y**4 + 6*x**12*y**6 + 12*x**8*y**10 + 12*x**6*y**12 + 6*x**2*y**16 + 6*y**18 - z**14*(x**4*(t**2*(t + 2)*(t*(4*t*(t + 4) + 17) + 10) + 3) + x**2*y**2*(2*t*(t*(t*(78 - 31*t) + 82) + 12) + 6) + y**4*(t*(t*(-t*(2*t - 1)*(t*(2*t + 13) + 28) + 148) + 24) + 15)) - z**10*(x**8*(t*(t*(t*(t*(t*(t*(13*t + 71) + 146) + 137) + 75) + 49) + 10) + 3) + x**6*y**2*(t*(t*(t*(t*(2*t*(t*(10 - 7*t) + 97) + 647) + 838) + 202) + 138) - 9) + x**4*y**4*(2*t*(t*(t*(-t*(t*(t*(62*t + 139) + 100) - 641) + 814) + 277) + 72) + 18) + x**2*y**6*(t*(t*(t*(-2*t**2*(t*(57*t + 134) + 23) + 359*t + 1222) + 782) + 30) + 51) + y**8*(t*(t*(t*(t*(t*(-t*(17*t + 57) + 2) + 87) + 25) + 449) - 30) + 45)) - z**8*(x**10*(t*(t*(t*(t*(t*(t*(17*t + 79) + 131) + 121) + 103) + 37) + 13) + 3) + x**8*y**2*(t*(t*(t*(t*(t*(t*(t + 55) + 496) + 923) + 549) + 469) + 6) + 21) + x**6*y**4*(t*(t*(t*(t*(t*(-4*t*(50*t + 147) + 1165) + 2246) + 1562) + 708) + 117) + 30) + x**4*y**6*(t*(t*(t*(t*(t*(-4*t*(71*t + 200) + 1299) + 2096) + 1590) + 968) + 183) - 12) + x**2*y**8*(t*(t*(t*(t*(-3*t*(19*t*(t + 3) - 68) + 1109) + 719) + 471) + 290) - 45) + y**10*(t*(t*(t*(t*(11*t**2*(t + 3) + t + 105) + 325) - 157) + 207) - 21)) - z**6*(t*x**12*(t*(t*(3*t*(t*(t*(3*t + 13) + 25) + 32) + 71) + 29) + 17) + 2*t*x**10*y**2*(t*(t*(t*(t*(t*(t + 58) + 246) + 226) + 331) + 110) + 36) + t*x**4*y**8*(t*(t*(t*(t*(t*(543 - 85*t) + 1167) + 1916) + 1105) + 289) + 105) + x**8*y**4*(t*(t*(t*(t*(t*(t*(185 - 207*t) + 1033) + 1990) + 1507) + 391) + 111) + 30) + x**6*y**6*(2*t*(t*(t*(t*(t*(t*(133 - 194*t) + 926) + 1317) + 858) + 243) + 62) + 30) + x**2*y**10*(2*t*(t*(t*(t*(t*(t*(41*t + 189) + 270) + 239) + 255) + 31) - 32) + 30) + y**12*(t*(t*(t*(t*(t*(t*(11*t + 73) + 185) + 74) - 67) + 171) - 141) + 30)) - z**4*(t*x**6*y**8*(t*(t*(t*(t*(t*(139*t + 1287) + 1948) + 1012) + 521) + 181) - 48) + x**14*(t**2*(t + 2)*(t*(t*(t*(t + 4) + 20) + 5) + 17) + 3) + x**12*y**2*(t*(t*(t*(t*(t*(9*t*(t + 6) + 146) + 377) + 203) + 150) + 78) - 9) + x**10*y**4*(t*(t*(t*(t*(t*(t*(41*t + 37) + 874) + 982) + 563) + 467) + 78) - 18) + x**8*y**6*(t*(t*(t*(t*(t*(t*(17*t + 657) + 1468) + 1670) + 835) + 283) + 128) - 18) + x**4*y**10*(t*(t*(t*(t*(t*(t*(179*t + 1189) + 1354) + 562) + 65) - 205) - 138) + 18) + x**2*y**12*(t*(t*(t*(t*(t*(t*(59*t + 326) + 858) + 325) - 271) - 186) - 82) - 21) + y**14*(t*(t*(t*(t*(t*(t*(3*t + 28) + 116) + 171) - 7) - 164) + 24) - 27)) - z**2*(t*x**16*(t**2*(t*(t + 8) + 20) + 7) + 2*t*x**8*y**8*(t*(t*(t*(t*(t*(132*t + 383) + 610) + 96) + 112) - 59) - 14) + t*y**16*(t*(2*t + 7)*(t*(t*(t + 8) + 8) - 6) - 63) + x**14*y**2*(t**2*(t*(t*(2*t*(t + 2)*(t + 4) + 87) + 106) + 50) + 15) + x**12*y**4*(t*(t*(t*(t*(t*(t*(25*t + 91) + 193) + 189) + 391) + 89) + 15) + 15) + x**10*y**6*(t*(t*(t*(t*(2*t*(32*t*(2*t + 7) + 127) + 809) + 296) + 62) - 2) + 21) + x**6*y**10*(t*(t*(t*(t*(2*t*(t*(100*t + 533) + 569) + 527) - 628) - 200) - 66) - 21) + x**4*y**12*(t*(t*(t*(t*(t*(t*(79*t + 437) + 1171) + 307) - 439) - 457) - 75) - 15) + x**2*y**14*(t*(t*(t*(t*(2*t*(t*(3*t + 29) + 184) + 433) - 206) - 280) - 76) - 15))) - (t*(x**2 + y**2) + z**2)*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**3)/(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**(7/2),
            y*z*(mu*(x**18*(6*t**4 + 16*t**3 - 12*t - 6) + x**16*(t*z**2*(t*(2*t + 7)*(t*(t*(t + 8) + 8) - 6) - 63) + y**2*(t*(t*(t*(t*(t*(2*t + 23) + 102) + 46) - 82) - 49) - 6)) + x**14*(t*y**4*(t*(t*(t*(3*t + 16)*(t*(t + 10) + 14) - 77) - 170) - 84) + y**2*z**2*(t*(t*(t*(t*(2*t*(t*(3*t + 29) + 184) + 433) - 206) - 280) - 76) - 15) + z**4*(t*(t*(t*(t*(t*(t*(3*t + 28) + 116) + 171) - 7) - 164) + 24) - 27)) + x**12*(y**6*(t*(t*(t*(t*(t*(t*(31*t + 184) + 406) + 156) - 173) - 204) - 52) - 12) + y**4*z**2*(t*(t*(t*(t*(t*(t*(79*t + 437) + 1171) + 307) - 439) - 457) - 75) - 15) + y**2*z**4*(t*(t*(t*(t*(t*(t*(59*t + 326) + 858) + 325) - 271) - 186) - 82) - 21) + z**6*(t*(t*(t*(t*(t*(t*(11*t + 73) + 185) + 74) - 67) + 171) - 141) + 30)) + x**10*(y**8*(t*(t*(t*(t*(t*(68*t**2 + 286*t + 355) + 84) - 130) - 98) - 49) - 12) + y**6*z**2*(t*(t*(t*(t*(2*t*(t*(100*t + 533) + 569) + 527) - 628) - 200) - 66) - 21) + y**4*z**4*(t*(t*(t*(t*(t*(t*(179*t + 1189) + 1354) + 562) + 65) - 205) - 138) + 18) + y**2*z**6*(2*t*(t*(t*(t*(t*(t*(41*t + 189) + 270) + 239) + 255) + 31) - 32) + 30) + z**8*(t*(t*(t*(t*(11*t**2*(t + 3) + t + 105) + 325) - 157) + 207) - 21)) + x**8*(t*y**10*(t*(t*(t*(t*(8*t*(8*t + 27) + 161) + 96) + 34) - 28) - 39) + 2*t*y**8*z**2*(t*(t*(t*(t*(t*(132*t + 383) + 610) + 96) + 112) - 59) - 14) + t*y**6*z**4*(t*(t*(t*(t*(t*(139*t + 1287) + 1948) + 1012) + 521) + 181) - 48) + t*y**4*z**6*(t*(t*(t*(t*(t*(543 - 85*t) + 1167) + 1916) + 1105) + 289) + 105) + y**2*z**8*(t*(t*(t*(t*(-3*t*(19*t*(t + 3) - 68) + 1109) + 719) + 471) + 290) - 45) + z**10*(t*(t*(t*(t*(t*(-t*(17*t + 57) + 2) + 87) + 25) + 449) - 30) + 45)) + x**6*(t*z**12*(-t*(t*(t*(t*(t*(4*t + 31) + 107) + 4) - 252) - 59) + 171) + y**12*(t*(t*(t*(t*(t*(t*(25*t + 76) + 82) + 54) + 81) + 16) + 8) - 6) + y**10*z**2*(t*(t*(t*(t*(2*t*(32*t*(2*t + 7) + 127) + 809) + 296) + 62) - 2) + 21) + y**8*z**4*(t*(t*(t*(t*(t*(t*(17*t + 657) + 1468) + 1670) + 835) + 283) + 128) - 18) + y**6*z**6*(2*t*(t*(t*(t*(t*(t*(133 - 194*t) + 926) + 1317) + 858) + 243) + 62) + 30) + y**4*z**8*(t*(t*(t*(t*(t*(-4*t*(71*t + 200) + 1299) + 2096) + 1590) + 968) + 183) - 12) + y**2*z**10*(t*(t*(t*(-2*t**2*(t*(57*t + 134) + 23) + 359*t + 1222) + 782) + 30) + 51)) + x**4*(t*y**2*z**12*(t*(t*(-t*(t*(t*(4*t + 53) + 375) - 288) + 632) + 361) + 159) + y**14*(t*(t*(t*(t*(t*(t*(t + 6) + 50) + 22) - 3) + 62) + 12) - 6) + y**12*z**2*(t*(t*(t*(t*(t*(t*(25*t + 91) + 193) + 189) + 391) + 89) + 15) + 15) + y**10*z**4*(t*(t*(t*(t*(t*(t*(41*t + 37) + 874) + 982) + 563) + 467) + 78) - 18) + y**8*z**6*(t*(t*(t*(t*(t*(t*(185 - 207*t) + 1033) + 1990) + 1507) + 391) + 111) + 30) + y**6*z**8*(t*(t*(t*(t*(t*(-4*t*(50*t + 147) + 1165) + 2246) + 1562) + 708) + 117) + 30) + y**4*z**10*(2*t*(t*(t*(-t*(t*(t*(62*t + 139) + 100) - 641) + 814) + 277) + 72) + 18) + z**14*(t*(t*(-t*(2*t - 1)*(t*(2*t + 13) + 28) + 148) + 24) + 15)) + x**2*(t*y**16*(t*(t + 2)*(t*(t + 6) + 2) + 9) + 2*t*y**10*z**6*(t*(t*(t*(t*(t*(t + 58) + 246) + 226) + 331) + 110) + 36) + t*y**4*z**12*(t*(t*(t*(t*(t*(4*t - 9) - 61) + 304) + 392) + 333) + 45) + 4*t*z**16*(t*(-2*t**2 + t + 4) + 6) + y**14*z**2*(t**2*(t*(t*(2*t*(t + 2)*(t + 4) + 87) + 106) + 50) + 15) + y**12*z**4*(t*(t*(t*(t*(t*(9*t*(t + 6) + 146) + 377) + 203) + 150) + 78) - 9) + y**8*z**8*(t*(t*(t*(t*(t*(t*(t + 55) + 496) + 923) + 549) + 469) + 6) + 21) + y**6*z**10*(t*(t*(t*(t*(2*t*(t*(10 - 7*t) + 97) + 647) + 838) + 202) + 138) - 9) + y**2*z**14*(2*t*(t*(t*(78 - 31*t) + 82) + 12) + 6)) + (y**2 + z**2)*(t*y**2 + z**2)*(y**4 + y**2*z**2*(t*(t + 2) - 1) + z**4)*(4*t**2*z**10 + 4*t*y**10 + 8*t*y**6*z**4*(t*(t + 1)**2 + 1) + 4*t*y**2*z**8*(t*(t + 1)**2 + 1) + y**8*z**2*(t**2*(t*(t + 4) + 12) + 3) + y**4*z**6*(t**2*(t*(9*t + 20) + 8) + 3))) + (t*(x**2 + y**2) + z**2)*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**3)/(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**(7/2),
            (x - y)*(x + y)*(mu*(-2*t**7*x**12*y**6 - 6*t**7*x**10*y**8 - 6*t**7*x**8*y**10 - 2*t**7*x**6*y**12 - t**6*x**14*y**4 - 15*t**6*x**12*y**6 - 48*t**6*x**10*y**8 - 48*t**6*x**8*y**10 - 15*t**6*x**6*y**12 - t**6*x**4*y**14 - 16*t**5*x**14*y**4 - 78*t**5*x**12*y**6 - 170*t**5*x**10*y**8 - 170*t**5*x**8*y**10 - 78*t**5*x**6*y**12 - 16*t**5*x**4*y**14 - 5*t**4*x**16*y**2 - 44*t**4*x**14*y**4 - 148*t**4*x**12*y**6 - 187*t**4*x**10*y**8 - 187*t**4*x**8*y**10 - 148*t**4*x**6*y**12 - 44*t**4*x**4*y**14 - 5*t**4*x**2*y**16 - 8*t**3*x**16*y**2 - 16*t**3*x**14*y**4 + 18*t**3*x**12*y**6 + 110*t**3*x**10*y**8 + 110*t**3*x**8*y**10 + 18*t**3*x**6*y**12 - 16*t**3*x**4*y**14 - 8*t**3*x**2*y**16 + 4*t**3*z**18 + 18*t**2*x**16*y**2 + 67*t**2*x**14*y**4 + 285*t**2*x**12*y**6 + 398*t**2*x**10*y**8 + 398*t**2*x**8*y**10 + 285*t**2*x**6*y**12 + 67*t**2*x**4*y**14 + 18*t**2*x**2*y**16 + 4*t**2*z**16*(t + 2)*(2*t**2 + 1)*(x**2 + y**2) + 4*t*x**18 + 16*t*x**16*y**2 + 124*t*x**14*y**4 + 210*t*x**12*y**6 + 326*t*x**10*y**8 + 326*t*x**8*y**10 + 210*t*x**6*y**12 + 124*t*x**4*y**14 + 16*t*x**2*y**16 + 4*t*y**18 + t*z**14*(x**4*(t*(t*(t*(t*(4*t*(t + 4) + 33) + 48) + 32) + 4) + 7) + x**2*y**2*(2*t**2*(t*(t*(2*t + 1)*(2*t + 7) + 90) + 16) + 22) + y**4*(t*(t*(t*(t*(4*t*(t + 4) + 33) + 48) + 32) + 4) + 7)) + t*z**10*(x**8*(t*(t*(t*(t*(t*(17*t + 83) + 147) + 128) + 79) + 33) + 17) + x**6*y**2*(t*(t*(t*(t*(4*t*(5*t + 37) + 627) + 688) + 166) + 368) - 1) + x**4*y**4*(2*t*(t*(t*(t*(t*(33 - 13*t) + 618) + 422) + 149) + 329) - 52) + x**2*y**6*(t*(t*(t*(t*(4*t*(5*t + 37) + 627) + 688) + 166) + 368) - 1) + y**8*(t*(t*(t*(t*(t*(17*t + 83) + 147) + 128) + 79) + 33) + 17)) + t*z**4*(x**2 + y**2)*(x**12*(t*(t*(t*(t*(t + 7) + 36) + 64) + 19) + 17) + x**10*y**2*(4*t*(-t*(t*(t*(t + 12) - 2) - 114) + 104) + 36) + x**8*y**4*(t*(t*(-t*(t*(t*(31*t + 193) + 806) - 820) + 1683) + 517) + 170) - x**6*y**6*(2*t*(t*(t*(t*(t*(29*t + 482) + 63) - 612) - 781) - 554) - 134) + x**4*y**8*(t*(t*(-t*(t*(t*(31*t + 193) + 806) - 820) + 1683) + 517) + 170) + x**2*y**10*(4*t*(-t*(t*(t*(t + 12) - 2) - 114) + 104) + 36) + y**12*(t*(t*(t*(t*(t + 7) + 36) + 64) + 19) + 17)) + 15*x**16*y**2 + 30*x**14*y**4 + 66*x**12*y**6 + 81*x**10*y**8 + 81*x**8*y**10 + 66*x**6*y**12 + 30*x**4*y**14 + 15*x**2*y**16 + z**12*(x**2 + y**2)*(x**4*(t**2*(t*(t*(t*(t*(13*t + 58) + 92) + 77) + 63) + 30) + 3) + x**2*y**2*(2*t*(t*(t*(t*(t*(-t*(t - 40) + 110) + 31) + 151) - 4) + 6) + 6) + y**4*(t**2*(t*(t*(t*(t*(13*t + 58) + 92) + 77) + 63) + 30) + 3)) + z**8*(x**2 + y**2)*(x**8*(t*(t*(t*(t*(t*(t*(9*t + 55) + 139) + 161) + 79) + 45) + 13) + 3) + x**6*y**2*(t*(t*(t*(t*(2*t*(t*(16 - 6*t) + 187) + 755) + 696) + 42) + 138) - 9) + x**4*y**4*(2*t*(t*(t*(t*(t*(-3*t*(35*t + 9) + 323) + 798) + 357) + 77) + 101) - 24) + x**2*y**6*(t*(t*(t*(t*(2*t*(t*(16 - 6*t) + 187) + 755) + 696) + 42) + 138) - 9) + y**8*(t*(t*(t*(t*(t*(t*(9*t + 55) + 139) + 161) + 79) + 45) + 13) + 3)) + z**6*(x**12*(t*(t*(t*(t*(t*(t*(t + 15) + 66) + 113) + 87) + 41) + 10) + 3) + x**10*y**2*(2*t*(t*(t*(t*(t*(t*(2*t - 3) + 20) + 384) + 396) + 189) + 8) + 24) + x**8*y**4*(t*(t*(t*(t*(t*(-t*(97*t + 723) + 340) + 2405) + 2205) + 807) + 28) + 75) + x**6*y**6*(4*t*(t*(t*(t*(t*(-t*(70*t + 327) + 217) + 835) + 726) + 267) + 5) + 108) + x**4*y**8*(t*(t*(t*(t*(t*(-t*(97*t + 723) + 340) + 2405) + 2205) + 807) + 28) + 75) + x**2*y**10*(2*t*(t*(t*(t*(t*(t*(2*t - 3) + 20) + 384) + 396) + 189) + 8) + 24) + y**12*(t*(t*(t*(t*(t*(t*(t + 15) + 66) + 113) + 87) + 41) + 10) + 3)) + z**2*(x**16*(t**2*(t*(t + 8) + 24) + 3) + x**14*y**2*(t*(t*(t*(-t*(9*t + 46) + 66) + 160) + 123) - 6) + x**12*y**4*(t*(-t*(t*(t*(t*(t*(3*t + 17) + 235) + 239) - 427) - 853) + 219) + 3) + x**10*y**6*(t*(-t*(t*(t*(t*(32*t*(t + 6) + 703) + 386) - 1534) - 1360) + 429) + 6) - x**8*y**8*(2*t*(t*(t*(t*(3*t*(t*(7*t + 69) + 167) - 62) - 741) - 931) - 261) + 12) + x**6*y**10*(t*(-t*(t*(t*(t*(32*t*(t + 6) + 703) + 386) - 1534) - 1360) + 429) + 6) + x**4*y**12*(t*(-t*(t*(t*(t*(t*(3*t + 17) + 235) + 239) - 427) - 853) + 219) + 3) + x**2*y**14*(t*(t*(t*(-t*(9*t + 46) + 66) + 160) + 123) - 6) + y**16*(t**2*(t*(t + 8) + 24) + 3))) + (t*z**2 + x**2 + y**2)*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**3)/(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**(7/2)
        ))

    # Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(phi)
    lset_approx = GridFunction(H1(mesh, order=1, dirichlet=[]))
    InterpolateToP1(phi, lset_approx)
    ci = CutInfo(mesh, lset_approx)

    ba_IF = ci.GetElementsOfType(IF)
    ba_IF_band = BitArray(mesh.ne)
    update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

    # FESpace: Taylor-Hood element
    V = VectorH1(mesh, order=order, dirichlet=[])
    Q = H1(mesh, order=order - 1, dirichlet=[])

    # define projection matrix
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ba_IF, deformation=deformation)
    dX = dx(definedonelements=ba_IF_band, deformation=deformation)
    dX2 = dx(definedonelements=ba_IF, deformation=deformation)

    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)
    h = specialcf.mesh_size
    tau = 1.0 / (h * h)
    rho_u = 1.0 / h
    rho_p = 1.0 * h

    gfu_prevs = [GridFunction(V) for i in range(time_order)]

    gfu_out = GridFunction(V)
    gfp_out = GridFunction(Q)
    if fname:
        filename = f"./vtk_out/diffusion/moving-ns-{fname}-mu={mu}"
    else:
        filename = f"./vtk_out/diffusion/moving-ns-mu={mu}"
    vtk = VTKOutput(mesh,
                    coefs=[lset_approx, phi, gfu_out, coef_u, gfp_out, coef_p],
                    names=["P1-levelset", "phi", "u", "uSol", "p", "pSol"],
                    filename=filename,
                    subdivision=0)

    for j in range(time_order):
        # fix levelset
        t.Set(-j * dt)
        t_curr = -j * dt

        deformation = lsetmeshadap.CalcDeformation(phi)

        # solve elliptic problem on a fixed surface to get u with normal extension
        InterpolateToP1(phi, lset_approx)
        ci = CutInfo(mesh, lset_approx)

        ba_IF.Clear()
        ba_IF |= ci.GetElementsOfType(IF)
        update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

        VG = Compress(V, GetDofsOfElements(V, ba_IF_band))

        # helper grid functions
        n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=phi, vel_space=V)

        gfu_el = GridFunction(VG)

        u_el, v_el = VG.TnT()

        a_el = BilinearForm(VG, symmetric=True)
        a_el += InnerProduct(Pmat * u_el, Pmat * v_el) * ds
        a_el += 2 * mu * (InnerProduct(Pmat * Sym(grad(u_el)) * Pmat - (u_el * n) * Hmat,
                                       Pmat * Sym(grad(v_el)) * Pmat - (v_el * n) * Hmat)) * ds
        a_el += (tau * ((u_el * n_k) * (v_el * n_k))) * ds
        a_el += (rho_u * InnerProduct(grad(u_el) * n, grad(v_el) * n)) * dX

        f_el = LinearForm(VG)
        f_el += InnerProduct(coef_f_el, Pmat * v_el) * ds

        with TaskManager():
            pre_a_el = Preconditioner(a_el, "bddc")

            a_el.Assemble()
            f_el.Assemble()

            solvers.CG(mat=a_el.mat, rhs=f_el.vec, pre=pre_a_el.mat, sol=gfu_el.vec, printrates=False)
            sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

            gfu_prevs[j].Set(gfu_el)

    # TIME MARCHING
    t.Set(0.0)
    t_curr = 0.0

    gfu_out.Set(gfu_prevs[0])
    gfp_out.Set(coef_p)
    vtk.Do(time=t_curr)

    i = 1

    l2err_old = 0.0

    while t_curr < tfinal - dt:
        t.Set(t_curr + dt)
        with TaskManager():
            deformation = lsetmeshadap.CalcDeformation(phi)

            InterpolateToP1(phi, lset_approx)
            ci = CutInfo(mesh, lset_approx)

            ba_IF.Clear()
            ba_IF |= ci.GetElementsOfType(IF)
            update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

            VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
            QG = Compress(Q, GetDofsOfElements(Q, ba_IF))
            XG = FESpace([VG, QG])
            u, p = XG.TrialFunction()
            v, q = XG.TestFunction()

            # helper grid functions
            n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=phi, vel_space=V)

        gfu_approx = GridFunction(VG)
        if time_order == 1:
            gfu_approx.Set(Pmat * gfu_prevs[0])
        elif time_order == 2:
            gfu_approx.Set(2 * Pmat * gfu_prevs[0] - Pmat * gfu_prevs[1])
        else:
            gfu_approx.Set(3 * Pmat * gfu_prevs[0] - 3 * Pmat * gfu_prevs[1] + Pmat * gfu_prevs[2])

        a = BilinearForm(XG, symmetric=False)
        a += bdf_coeff[0] / dt * InnerProduct(u, Pmat * v) * ds
        a += wN * InnerProduct(Hmat * u, Pmat * v) * ds
        a += 0.5 * InnerProduct((Pmat * grad(u) * Pmat - (u * n) * Hmat) * gfu_approx, v) * ds
        a += (-0.5) * InnerProduct((Pmat * grad(v) * Pmat - (v * n) * Hmat) * gfu_approx, u) * ds
        a += (-0.5) * InnerProduct(coef_g * u, Pmat * v) * ds
        a += 2.0 * mu * (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                                      Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        a += tau * InnerProduct(n_k, u) * InnerProduct(n_k, v) * ds
        a += rho_u * InnerProduct(grad(u) * n, grad(v) * n) * dX

        a += InnerProduct(u, Pmat * grad(q)) * ds
        a += InnerProduct(v, Pmat * grad(p)) * ds

        a += (-1.0) * rho_p * (grad(p) * n) * (grad(q) * n) * dX2

        f = LinearForm(XG)
        f += InnerProduct(coef_f, Pmat * v) * ds
        for j in range(time_order):
            f += (-1.0) * bdf_coeff[j + 1] / dt * InnerProduct(gfu_prevs[j], Pmat * v) * ds
        f += (-1.0) * coef_g * q * ds

        with TaskManager():
            a.Assemble()
            f.Assemble()

            gf = GridFunction(XG)

        gf.vec.data = a.mat.Inverse(freedofs=XG.FreeDofs(), inverse="umfpack") * f.vec

        gfu = gf.components[0]

        l2err = sqrt(Integrate(InnerProduct(Pmat * (gfu - gfu_prevs[0]), Pmat * (gfu - gfu_prevs[0])) * ds, mesh))

        if i > 1 and l2err > 2 * l2err_old:
            continue

        for j in range(time_order-1):
            gfu_prevs[-1 - j].vec.data = gfu_prevs[-2 - j].vec

        gfp = gf.components[1]
        # making numerical pressure mean zero
        renormalize(QG, mesh, ds, gfp)

        gfu_out.Set(gfu)
        gfp_out.Set(gfp)
        vtk.Do(time=t_curr+dt)

        gfu_prevs[0].Set(gfu)

        print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        l2err_old = l2err
        t_curr += dt
        i += 1

    return

bbox_sz = 2.0
min_unif_ref = 4
max_unif_ref = 5

sns.set()
tfinal = 1.0

print("\r", f"    h     :   l2is   ,   l2us   ,   h1us   ,   l2ps   ,   h1ps              ")

test_name = "l2l4-nonsol-demo"
plt_out = False

for unif_ref in range(min_unif_ref, max_unif_ref):
    # h, dt, ts, l2us, h1us, l2ps, h1ps = moving_ns_direct(
    #     order=2, unif_ref=unif_ref, bbox_sz=bbox_sz,
    #     tfinal=tfinal, time_order=2, out=True,
    #     fname=test_name, test_name=test_name
    # )
    #
    # l2 = max(l2us)
    # l2u = np.sqrt(sci.simps(y=np.array(l2us) ** 2, x=ts, dx=dt, even='avg'))
    # h1u = np.sqrt(sci.simps(y=np.array(h1us) ** 2, x=ts, dx=dt, even='avg'))
    # l2p = np.sqrt(sci.simps(y=np.array(l2ps) ** 2, x=ts, dx=dt, even='avg'))
    # h1p = np.sqrt(sci.simps(y=np.array(h1ps) ** 2, x=ts, dx=dt, even='avg'))
    #
    # if plt_out:
    #     fig, axs = plt.subplots(2, 2)
    #
    #     fig.set_figwidth(20)
    #     fig.set_figheight(15)
    #
    #     plt.suptitle(rf"h={h:.3E}, dt=h/4")
    #
    #     axs[0, 0].plot(ts, l2us)
    #     axs[0, 0].set_title(rf"$L^2$-error in $\mathbf{{u}}$")
    #     axs[0, 0].set_ylim(0.0)
    #
    #     axs[0, 1].plot(ts, h1us)
    #     axs[0, 1].set_title(rf"$H^1$-error in $\mathbf{{u}}$")
    #     axs[0, 1].set_ylim(0.0)
    #
    #     axs[1, 0].plot(ts, l2ps)
    #     axs[1, 0].set_title(rf"$L^2$-error in $p$")
    #     axs[1, 0].set_ylim(0.0)
    #
    #     axs[1, 1].plot(ts, h1ps)
    #     axs[1, 1].set_title(rf"$H^1$-error in $p$")
    #     axs[1, 1].set_ylim(0.0)
    #
    #     plt.savefig(f"./figures/moving_ns/translation-{test_name}-h={h}.png")
    #
    # print("\r", f"{h:.3E} : {l2:.3E}, {l2u:.3E}, {h1u:.3E}, {l2p:.3E}, {h1p:.3E}              ")

    moving_ns_for_demos(
        order=2, unif_ref=unif_ref, bbox_sz=bbox_sz,
        tfinal=tfinal, time_order=1,
        fname=test_name, test_name=test_name
    )


# Non-solonoidal BDF2: P2/P1
#     h: l2is, l2us, h1us, l2ps, h1ps
# 5.000E-01: 2.360E-01, 1.791E-01, 4.975E-01, 3.198E-01, 1.322E+00
# 2.500E-01: 9.141E-03, 6.677E-03, 8.681E-02, 4.749E-02, 6.083E-01
# 1.250E-01: 1.957E-03, 1.382E-03, 2.291E-02, 8.883E-03, 2.988E-01
# 6.250E-02: 4.849E-04, 3.382E-04, 6.676E-03, 2.030E-03, 1.532E-01

# l4l2 incompressible: BDF2: P2/P1
#     h     :   l2is   ,   l2us   ,   h1us   ,   l2ps   ,   h1ps
# 5.000E-01 : 7.826E-01, 3.538E-01, 9.862E-01, 5.700E-01, 2.029E+00
# 2.500E-01 : 2.306E-01, 3.315E-02, 2.485E-01, 9.082E-02, 8.815E-01
# 1.250E-01 : 1.237E-02, 7.951E-03, 6.470E-02, 1.998E-02, 4.295E-01
# 6.250E-02 : 1.781E-03, 9.733E-04, 1.735E-02, 3.827E-03, 2.142E-01
