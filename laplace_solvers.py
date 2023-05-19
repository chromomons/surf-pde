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


def diffusion(mesh, dt, tfinal=1.0, order=1, out=False, stab_type='old', bad_rhs=False, **exact):
    if order < 3:
        precond_name = "bddc"
        cg_iter = 10
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

    alpha = 2.0 - np.sqrt(2.0)

    t = Parameter(0.0)
    tfun = 1. + sin(pi*t)
    tfun_dif = tfun.Diff(t)

    # declare grid functions to store the solution
    gfu = GridFunction(V)

    # declare the integration domains
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ci.GetElementsOfType(IF), deformation=deformation)
    dX = dx(definedonelements=ci.GetElementsOfType(IF), deformation=deformation)

    uSol = CoefficientFunction(tfun*exact["u"]).Compile()
    rhsf = CoefficientFunction(tfun*exact["f"] + tfun_dif*exact["u"]).Compile()

    # define projection matrix
    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)

    ### bilinear forms:
    bilinear_form_args = {'alpha': alpha, 'dt': dt, 'stab_type': stab_type}
    m, d, a, f = define_forms(eq_type='total_stab_tests_diffusion', V=V, n=n, Pmat=Pmat, rhsf=rhsf, ds=ds, dX=dX, **bilinear_form_args)

    start = time.perf_counter()
    with TaskManager():
        prea = Preconditioner(a, precond_name)
        assemble_forms([m, d, a, f])

    print(f"{bcolors.OKCYAN}System assembled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    ### LINEAR SOLVER

    diff = gfu.vec.CreateVector()
    fold = f.vec.CreateVector()
    rhs = f.vec.CreateVector()

    # TIME MARCHING

    t_curr = 0.0  # time counter within one block-run

    # IC
    t.Set(0.0)

    if bad_rhs:
        gfu_el = GridFunction(V)
        rhsf_el = CoefficientFunction(exact["f"] + exact["u"]).Compile()
        bilinear_form_args = {'mass_cf': 1.0}
        a_el, f_el = define_forms(eq_type='poisson', V=V, n=n, Pmat=Pmat, rhsf=rhsf_el, ds=ds, dX=dX, **bilinear_form_args)
        start = time.perf_counter()
        with TaskManager():
            prea_el = Preconditioner(a_el, precond_name)
            assemble_forms([a_el, f_el])
            solvers.CG(mat=a_el.mat, pre=prea_el.mat, rhs=f_el.vec, sol=gfu_el.vec, maxsteps=cg_iter, initialize=True, tol=1e-12,
                       printrates=False)
        sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning
        gfu.Set(tfun * gfu_el)
        print(f"{bcolors.OKGREEN}IC computed      ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")
    else:
        mesh.SetDeformation(deformation)
        gfu.Set(uSol)
        mesh.UnsetDeformation()

    if out:
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, deformation, gfu, uSol],
                        names=["P1-levelset", "deform", "u", "uSol"],
                        filename=f"./vtk_out/diffusion/{exact['name']}",
                        subdivision=0)
        vtk.Do(time=0.0)

    out_errs = {'ts': [], 'l2us': [], 'h1us': []}

    with TaskManager():
        l2u, h1u = errors(mesh, ds, Pmat, gfu, uSol)
    append_errors(t_curr, l2u, h1u, **out_errs)

    start = time.perf_counter()

    fold.data = f.vec
    i = 0
    while t_curr < tfinal - 0.5 * dt:
        t.Set(t_curr + alpha*dt)
        with TaskManager():
            f.Assemble()

        # TR
        rhs.data = fold + f.vec - 2 * d.mat * gfu.vec

        with TaskManager():
            solvers.CG(mat=a.mat, pre=prea.mat, rhs=rhs, sol=diff, initialize=True, maxsteps=cg_iter, tol=1e-12,
                           printrates=False)
            gfu.vec.data += diff
        sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

        # BDF2
        t.Set(t_curr + dt)
        with TaskManager():
            f.Assemble()

        rhs.data = f.vec + (1.0-alpha)/(alpha*dt)* m.mat * diff - d.mat * gfu.vec

        with TaskManager():
            solvers.CG(mat=a.mat, pre=prea.mat, rhs=rhs, sol=diff, initialize=True, maxsteps=cg_iter, tol=1e-12,
                           printrates=False)
            gfu.vec.data += diff
        sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

        t_curr += dt

        fold.data = f.vec

        print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        with TaskManager():
            l2u, h1u = errors(mesh, ds, Pmat, gfu, uSol)
        append_errors(t_curr, l2u, h1u, **out_errs)

        if out:
            with TaskManager():
                vtk.Do(time=t_curr)
        i += 1

    print("")
    end = time.perf_counter()
    print(f" Time elapsed: {end - start: .5f} s")

    return V.ndof, out_errs['ts'], out_errs['l2us'], out_errs['h1us']
