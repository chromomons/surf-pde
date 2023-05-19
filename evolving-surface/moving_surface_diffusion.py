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
def errors(mesh, ds, Pmat, gfu, uSol):
    return sqrt(Integrate(InnerProduct(gfu - uSol, gfu - uSol) * ds, mesh=mesh)),\
           sqrt(Integrate(InnerProduct(Pmat * (grad(gfu) - coef_fun_grad(uSol)), Pmat * (grad(gfu) - coef_fun_grad(uSol))) * ds, mesh=mesh))


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


def append_errors(t_curr, l2u, h1u, **errs):
    errs['ts'].append(t_curr)
    errs['l2us'].append(l2u)
    errs['h1us'].append(h1u)


# SOLVERS
def poisson(mesh, mass_cf=1.0, order=1, out=False, **exact):
    if order < 3:
        precond_name = "bddc"
        cg_iter = 5
    else:
        precond_name = "local"
        cg_iter = 100000

    phi = CoefficientFunction(exact["phi"]).Compile()

    # LEVELSET ADAPTATION
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(phi)
    lset_approx = lsetmeshadap.lset_p1
    ci = CutInfo(mesh, lset_approx)

    # FESpace
    Vh = H1(mesh, order=order, dirichlet=[])
    V = Compress(Vh, GetDofsOfElements(Vh, ci.GetElementsOfType(IF)))

    # declare grid functions to store the solution
    gfu = GridFunction(V)

    # declare integration domains
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ci.GetElementsOfType(IF), deformation=deformation)
    dX = dx(definedonelements=ci.GetElementsOfType(IF), deformation=deformation)

    # unpacking exact solution
    uSol = CoefficientFunction(exact["u"]).Compile()
    rhsf = CoefficientFunction(exact["f"]).Compile()

    # define projection matrix
    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)

    # bilinear forms:
    bilinear_form_args = {'mass_cf': mass_cf}
    a, f = define_forms(eq_type='poisson', V=V, n=n, Pmat=Pmat, rhsf=rhsf, ds=ds, dX=dX, **bilinear_form_args)

    start = time.perf_counter()
    with TaskManager():
        prea = Preconditioner(a, precond_name)
        assemble_forms([a, f])

    print(f"{bcolors.OKCYAN}System assembled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    # LINEAR SOLVER

    start = time.perf_counter()
    with TaskManager():
        solvers.CG(mat=a.mat, pre=prea.mat, rhs=f.vec, sol=gfu.vec, maxsteps=cg_iter, initialize=True, tol=1e-12, printrates=False)
    sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning
    print(f"{bcolors.OKBLUE}System solved    ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    # ERRORS

    with TaskManager():
        l2u, h1u = errors(mesh, ds, Pmat, gfu, uSol)

    if out:
        with TaskManager():
            vtk = VTKOutput(ma=mesh,
                            coefs=[lset_approx, deformation, gfu, uSol],
                            names=["P1-levelset", "deform", "u", "uSol"],
                            filename=f"poisson", subdivision=0)
            vtk.Do()

    return V.ndof, l2u, h1u


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


class Exact:
    def __init__(self, nu, R, maxvel):
        self.t = Parameter(0.0)
        self.nu = Parameter(nu)
        self.R = Parameter(R)
        self.maxvel = Parameter(maxvel)

        self.phi = None
        self.w = None
        self.u = None
        self.f = None
        self.g = None
        self.fel = None
        self.divGw = None
        self.divGwT = None

    def set_params(self, phi, w, u, f, fel, divGw, divGwT):
        self.phi = phi
        self.w = w
        self.u = u
        self.f = f
        self.fel = fel
        self.divGw = divGw
        self.divGwT = divGwT

    def set_time(self, tval):
        self.t.Set(tval)


def moving_diffusion(order, unif_ref, bbox_sz, tfinal, exact_sol_type="translation", time_order=1, out=False):
    bdf_coeffs = [[1, -1],
                  [3 / 2, -2, 1 / 2],
                  [11 / 6, -3, 3 / 2, -1 / 3]]

    bdf_coeff = bdf_coeffs[time_order-1]

    c_delta = time_order + 0.1
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
        w3 = CoefficientFunction(-z*(t*(x**2 + y**2) + z**2)*(x**2*(y**2 + z**2) + y**2*z**2)/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)))
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
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order, heapsize=100000000)
    defold = GridFunction(V)
    deformation = lsetmeshadap.CalcDeformation(phi)

    lset_approx = GridFunction(H1(mesh, order=1, dirichlet=[]))
    InterpolateToP1(phi, lset_approx)
    ci = CutInfo(mesh, lset_approx)

    ba_IF = ci.GetElementsOfType(IF)
    ba_IF_band = BitArray(mesh.ne)
    update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

    # define projection matrix
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ba_IF, deformation=deformation)
    dX = dx(definedonelements=ba_IF_band, deformation=deformation)

    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)
    h = specialcf.mesh_size
    rho = 1./h

    gfu_prevs = [GridFunction(V, name=f"gru-{i}") for i in range(time_order)]

    if out:
        gfu_out = GridFunction(V)
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, gfu_out, coef_u],
                        names=["P1-levelset", "u", "uSol"],
                        filename=f"./vtk_out/diffusion/moving-diff-new",
                        subdivision=0)

    out_errs = {'ts': [], 'l2us': [], 'h1us': []}

    # IC
    for j in range(time_order):
        # fix levelset
        t.Set(-j*dt)
        deformation = lsetmeshadap.CalcDeformation(phi)
        t_curr = -j * dt

        InterpolateToP1(phi, lset_approx)
        ci = CutInfo(mesh, lset_approx)

        ba_IF.Clear()
        ba_IF |= ci.GetElementsOfType(IF)
        update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

        # solve elliptic problem on a fixed surface to get u with normal extension

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

            gfu_prevs[j].Set(gfu_el)

        if out:
            gfu_out.Set(gfu_prevs[j])
            vtk.Do(time=t_curr)

        with TaskManager():
            l2u, h1u = errors(mesh, ds, Pmat, gfu_prevs[j], coef_u)
        append_errors(t_curr, l2u, h1u, **out_errs)

    # TIME MARCHING

    t.Set(0.0)
    t_curr = 0.0

    i = 1

    while t_curr < tfinal + dt/2:
        t.Set(t_curr + dt)
        t_curr += dt
        with TaskManager():
            defold.vec.data = deformation.vec
            deformation = lsetmeshadap.CalcDeformation(phi)
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

            gfu = GridFunction(VG)

            solvers.GMRes(A=a.mat, b=f.vec, pre=c.mat, x=gfu.vec, tol=1e-15, maxsteps=15, printrates=False)

            for j in range(time_order-1):
                gfu_prevs[-1-j].vec.data = gfu_prevs[-2-j].vec
            gfu_prevs[0].Set(gfu)

            if out:
                gfu_out.Set(gfu)
                vtk.Do(time=t_curr)

        with TaskManager():
            l2u, h1u = errors(mesh, ds, Pmat, gfu, coef_u)
        append_errors(t_curr, l2u, h1u, **out_errs)

        print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        i += 1

    return h_approx, dt, out_errs['ts'], out_errs['l2us'], out_errs['h1us']


def moving_diffusion_new(mesh, dt, order, tfinal, exact, time_order=1, out=False):
    bdf_coeffs = [[1, -1],
                  [3 / 2, -2, 1 / 2],
                  [11 / 6, -3, 3 / 2, -1 / 3]]

    bdf_coeff = bdf_coeffs[time_order-1]

    c_delta = time_order + 0.1
    # MESH

    V = H1(mesh, order=order, dirichlet=[])

    # LEVELSET ADAPTATION
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(exact.phi)

    lset_approx = GridFunction(H1(mesh, order=1, dirichlet=[]))
    InterpolateToP1(exact.phi, lset_approx)
    ci = CutInfo(mesh, lset_approx)

    ba_IF = ci.GetElementsOfType(IF)
    ba_IF_band = BitArray(mesh.ne)
    update_ba_IF_band(lset_approx, mesh, c_delta * dt * exact.maxvel, ba_IF_band)

    # define projection matrix
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ba_IF, deformation=deformation)
    dX = dx(definedonelements=ba_IF_band, deformation=deformation)

    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)
    h = specialcf.mesh_size
    rho = 1./h

    gfu_prevs = [GridFunction(V, name=f"gru-{i}") for i in range(time_order)]

    if out:
        gfu_out = GridFunction(V)
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, gfu_out, exact.u],
                        names=["P1-levelset", "u", "uSol"],
                        filename=f"./vtk_out/diffusion/moving-diff-new",
                        subdivision=0)

    out_errs = {'ts': [], 'l2us': [], 'h1us': []}

    # IC
    for j in range(time_order):
        # fix levelset
        exact.set_time(-j*dt)
        deformation = lsetmeshadap.CalcDeformation(exact.phi)
        t_curr = -j * dt

        InterpolateToP1(exact.phi, lset_approx)
        ci = CutInfo(mesh, lset_approx)

        ba_IF.Clear()
        ba_IF |= ci.GetElementsOfType(IF)
        update_ba_IF_band(lset_approx, mesh, c_delta * dt * exact.maxvel, ba_IF_band)

        # solve elliptic problem on a fixed surface to get u with normal extension

        VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
        gfu_el = GridFunction(VG)
        u_el, v_el = VG.TnT()

        a_el = BilinearForm(VG, symmetric=True)
        a_el += (u_el*v_el + InnerProduct(Pmat * grad(u_el), Pmat * grad(v_el))) * ds
        a_el += 1./h * (n * grad(u_el)) * (n * grad(v_el)) * dX

        f_el = LinearForm(VG)
        f_el += exact.fel * v_el * ds

        with TaskManager():
            c_el = Preconditioner(a_el, "bddc")
            a_el.Assemble()
            f_el.Assemble()

            solvers.CG(mat=a_el.mat, rhs=f_el.vec, pre=c_el.mat, sol=gfu_el.vec, printrates=False)
            sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

            gfu_prevs[j].Set(gfu_el)

        if out:
            gfu_out.Set(gfu_prevs[j])
            vtk.Do(time=t_curr)

        with TaskManager():
            l2u, h1u = errors(mesh, ds, Pmat, gfu_prevs[j], exact.u)
        append_errors(t_curr, l2u, h1u, **out_errs)

    # TIME MARCHING

    dofs = []

    exact.set_time(0.0)
    t_curr = 0.0

    i = 1

    while t_curr < tfinal + dt/2:
        exact.set_time(t_curr + dt)
        with TaskManager():
            deformation = lsetmeshadap.CalcDeformation(exact.phi)
            InterpolateToP1(exact.phi, lset_approx)
            ci = CutInfo(mesh, lset_approx)

            ba_IF.Clear()
            ba_IF |= ci.GetElementsOfType(IF)
            update_ba_IF_band(lset_approx, mesh, c_delta * dt * exact.maxvel, ba_IF_band)

            VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
            dofs.append(VG.ndof)
            u, v = VG.TnT()

        a = BilinearForm(VG)
        a += (bdf_coeff[0] * u * v +
                     dt * (1. / 2 * InnerProduct(Pmat * exact.w, Pmat * grad(u)) * v -
                           1. / 2 * InnerProduct(Pmat * exact.w, Pmat * grad(v)) * u +
                           (exact.divGw - 0.5 * exact.divGwT) * u * v +
                           exact.nu * InnerProduct(Pmat * grad(u), Pmat * grad(v)))) * ds
        a += (dt * rho * (n * grad(u)) * (n * grad(v))) * dX

        f = LinearForm(VG)
        f += (dt * exact.f - sum([bdf_coeff[j+1] * gfu_prevs[j] for j in range(time_order)])) * v * ds

        with TaskManager():
            c = Preconditioner(a, "bddc")
            a.Assemble()
            f.Assemble()

            gfu = GridFunction(VG)

            solvers.GMRes(A=a.mat, b=f.vec, pre=c.mat, x=gfu.vec, tol=1e-15, maxsteps=15, printrates=False)

            for j in range(time_order-1):
                gfu_prevs[-1-j].vec.data = gfu_prevs[-2-j].vec
            gfu_prevs[0].Set(gfu)

            if out:
                gfu_out.Set(gfu)
                vtk.Do(time=t_curr)

        print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        t_curr += dt

        with TaskManager():
            l2u, h1u = errors(mesh, ds, Pmat, gfu, exact.u)
        append_errors(t_curr, l2u, h1u, **out_errs)

        i += 1
    print("")

    return np.mean(dofs), out_errs['ts'], out_errs['l2us'], out_errs['h1us']


# exact_sol_type = 'l4l2-sphere'
# # exact_sol_type = 'translation'
#
# if exact_sol_type == 'translation':
#     bbox_sz = 2.0
#     min_unif_ref = 3
#     max_unif_ref = 8
# else:
#     bbox_sz = 2.0
#     min_unif_ref = 3
#     max_unif_ref = 8
#
# sns.set()
# tfinal = 1.0
#
# print("\r", f"    h     :   l2l2   ,   lil2   ,   l2h1              ")
#
# for unif_ref in range(min_unif_ref, max_unif_ref):
#     h, dt, ts, l2us, h1us = moving_diffusion(order=1, unif_ref=unif_ref, bbox_sz=bbox_sz, tfinal=tfinal,
#                                              time_order=2, out=False, exact_sol_type=exact_sol_type)
#     lil2 = max(l2us)
#     l2l2 = np.sqrt(sci.simps(y=np.array(l2us)**2, x=ts, dx=dt, even='avg'))
#     l2h1 = np.sqrt(sci.simps(y=np.array(h1us)**2, x=ts, dx=dt, even='avg'))
#     plt.plot(l2us)
#     plt.show()
#     plt.plot(h1us)
#     plt.show()
#     print("\r", f"{h:.3E} : {l2l2:.3E}, {lil2:.3E}, {l2h1:.3E}              ")