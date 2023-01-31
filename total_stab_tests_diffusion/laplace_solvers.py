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
import scipy.sparse as sp
import scipy.sparse.linalg as lg


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
    elif eq_type == 'new_stab_tests-adaptive-be':
        # penalization parameters
        stab_type = args['stab_type']

        m = BilinearForm(V, symmetric=True)
        a = BilinearForm(V, symmetric=True)

        # mass part
        m += u * v * ds

        if stab_type in ['new', 'total']:
            # stabilizing mass part
            m += h * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX

        if stab_type in ['old', 'total']:
            # stabilizing new_stab_tests part
            a += 1.0 / h * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX

        # new_stab_tests part
        a += InnerProduct(Pmat * grad(u), Pmat * grad(v)) * ds

        f = LinearForm(V)
        f += rhsf * v * ds

        return m, a, f
    else:
        # penalization parameters
        rho_u = 1.0 / h
        alpha = args['alpha']
        dt = args['dt']
        stab_type = args['stab_type']

        m = BilinearForm(V, symmetric=True)  # mass
        d = BilinearForm(V, symmetric=True)  # new_stab_tests
        a = BilinearForm(V, symmetric=True)  # mass-new_stab_tests

        # mass part
        m += u * v * ds
        a += 2.0 / (alpha * dt) * u * v * ds

        if stab_type in ['new', 'total']:
            # stabilizing mass part
            m += h * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX
            a += 2.0 / (alpha * dt) * h * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX

        if stab_type in ['old', 'total']:
            # stabilizing new_stab_tests part
            d += rho_u * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX
            a += rho_u * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX

        # new_stab_tests part
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
    m, d, a, f = define_forms(eq_type='new_stab_tests', V=V, n=n, Pmat=Pmat, rhsf=rhsf, ds=ds, dX=dX, **bilinear_form_args)

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


def diffusion_adaptive(mesh, tfinal=1.0, order=1, out=False, stab_type='old', bad_rhs=False, **exact):
    dt = tfinal / 2
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

    t = Parameter(0.0)
    ntang = 10
    tfun = IfPos(t - 8 / ntang, 2 + sqrt(1 - (ntang * t - 9) ** 2),
                 IfPos(t - 6 / ntang, 2 - sqrt(1 - (ntang * t - 7) ** 2),
                       IfPos(t - 4 / ntang, 2 + sqrt(1 - (ntang * t - 5) ** 2),
                             IfPos(t - 2 / ntang, 2 - sqrt(1 - (ntang * t - 3) ** 2),
                                   2 + sqrt(1 - (ntang * t - 1) ** 2))
                             )
                       )
                 )
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
    bilinear_form_args = {'dt': dt, 'stab_type': stab_type}
    m, a, f = define_forms(eq_type='new_stab_tests-adaptive-be', V=V, n=n, Pmat=Pmat, rhsf=rhsf, ds=ds, dX=dX, **bilinear_form_args)

    start = time.perf_counter()
    with TaskManager():
        assemble_forms([m, a, f])

    print(f"{bcolors.OKCYAN}System assembled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    ### LINEAR SOLVER

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
    tol = 0.1 * l2u

    dts = []
    cg_iters = []

    gfu1 = GridFunction(V)
    gfu2 = GridFunction(V)
    rhs1 = f.vec.CreateVector()
    rhs2 = f.vec.CreateVector()

    mstar1 = m.mat.CreateMatrix()
    mstar2 = m.mat.CreateMatrix()
    freedofs = V.FreeDofs()
    preI = Projector(mask=freedofs, range=True)

    gfu1.Set(uSol)
    gfu2.Set(uSol)

    start = time.perf_counter()

    i = 0
    while t_curr < tfinal - 0.5 * dt:
        cg_it = 0

        gfu1.vec.data = gfu.vec
        gfu2.vec.data = gfu.vec

        with TaskManager():
            mstar1.AsVector().data = m.mat.AsVector() + dt * a.mat.AsVector()
            mstar2.AsVector().data = m.mat.AsVector() + dt / 2 * a.mat.AsVector()

        t.Set(t_curr + dt / 2)
        with TaskManager():
            f.Assemble()

        # first half-step
        rhs2.data = dt / 2.0 * (f.vec - a.mat * gfu2.vec)

        with TaskManager():
            diff2, cg_iter_num = solvers.CG(mat=mstar2, pre=preI, rhs=rhs2, initialize=True, maxsteps=cg_iter, tol=1e-12, printrates=False)
            gfu2.vec.data += diff2
            cg_it += cg_iter_num
        sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

        # second half-step
        t.Set(t_curr + dt)
        with TaskManager():
            f.Assemble()

        rhs1.data = dt * (f.vec - a.mat * gfu1.vec)
        rhs2.data = dt / 2 * (f.vec - a.mat * gfu2.vec)

        with TaskManager():
            diff1, cg_iter_num = solvers.CG(mat=mstar1, pre=preI, rhs=rhs1, initialize=True, maxsteps=cg_iter, tol=1e-12, printrates=False)
            sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning
            gfu1.vec.data += diff1
            cg_it += cg_iter_num

            diff2, cg_iter_num = solvers.CG(mat=mstar2, pre=preI, rhs=rhs2, initialize=True, maxsteps=cg_iter, tol=1e-12, printrates=False)
            sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning
            gfu2.vec.data += diff2
            cg_it += cg_iter_num

            l2_est = sqrt(Integrate(InnerProduct(gfu2 - gfu1, gfu2 - gfu1) * ds, mesh=mesh))

        if l2_est < tol:
            gfu.vec.data = 2 * gfu2.vec - gfu1.vec
            t_curr += dt

            print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

            with TaskManager():
                l2u, h1u = errors(mesh, ds, Pmat, gfu, uSol)
            append_errors(t_curr, l2u, h1u, **out_errs)

            dts.append(dt)
            cg_iters.append(cg_it)
            i += 1

        # https://en.wikipedia.org/wiki/Adaptive_step_size#Example
        dt = float(0.9 * dt * min(max((tol / (2 * l2_est)) ** (1. / 2), 0.3), 2))

        if out:
            with TaskManager():
                vtk.Do(time=t_curr)

    print("")
    end = time.perf_counter()
    print(f" Time elapsed: {end - start: .5f} s")

    return V.ndof, out_errs['ts'], out_errs['l2us'], out_errs['h1us'], dts, cg_iters


def compute_cond(mesh, dts, ref_lvl, sparse_cond=False, **exact):
    order = 1
    phi = CoefficientFunction(exact["phi"]).Compile()

    # LEVELSET ADAPTATION
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order + 1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(phi)
    lset_approx = lsetmeshadap.lset_p1

    ci = CutInfo(mesh, lset_approx)

    # FESpace
    Vh = H1(mesh, order=order, dirichlet=[])
    V = Compress(Vh, GetDofsOfElements(Vh, ci.GetElementsOfType(IF)))

    # declare the integration domains
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ci.GetElementsOfType(IF), deformation=deformation)
    dX = dx(definedonelements=ci.GetElementsOfType(IF), deformation=deformation)

    # define projection matrix
    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)

    # bilinear forms:
    u, v = V.TnT()
    h = specialcf.mesh_size

    # penalization parameters
    mo = BilinearForm(V, symmetric=True)
    mn = BilinearForm(V, symmetric=True)
    mt = BilinearForm(V, symmetric=True)
    ao = BilinearForm(V, symmetric=True)
    an = BilinearForm(V, symmetric=True)
    at = BilinearForm(V, symmetric=True)

    # mass part
    mo += u * v * ds
    mn += u * v * ds
    mt += u * v * ds

    # stabilizing mass part
    mn += h * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX
    mt += h * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX

    # stabilizing new_stab_tests part
    ao += 1.0 / h * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX
    at += 1.0 / h * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX

    # new_stab_tests part
    ao += InnerProduct(Pmat * grad(u), Pmat * grad(v)) * ds
    an += InnerProduct(Pmat * grad(u), Pmat * grad(v)) * ds
    at += InnerProduct(Pmat * grad(u), Pmat * grad(v)) * ds

    start = time.perf_counter()
    with TaskManager():
        assemble_forms([mo, mn, mt, ao, an, at])

    mats = {"old": [mo, ao], "new": [mn, an], "total": [mt, at]}

    print(f"{bcolors.OKCYAN}System assembled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    mstaro = mo.mat.CreateMatrix()
    mstarn = mn.mat.CreateMatrix()
    mstart = mt.mat.CreateMatrix()

    mstars = {"old": mstaro, "new": mstarn, "total": mstart}

    conds = {}

    start = time.perf_counter()

    for mode in ["old", "new", "total"]:
        with TaskManager():
            dt = dts[mode]
            mstar = mstars[mode]
            m = mats[mode][0]
            a = mats[mode][1]
            mstar.AsVector().data = m.mat.AsVector() + dt * a.mat.AsVector()

            rows, cols, vals = mstar.COO()
            A = sp.csr_matrix((vals, (rows, cols)))

            if sparse_cond:
                ew1, ev = lg.eigsh(A, which='LM')
                # ew2, ev = lg.eigsh(A, sigma=1e-12, k=100*2**ref_lvl, ncv=400*2**ref_lvl)  # <--- takes a long time
                ew2, ev = lg.eigsh(A, sigma=1e-8)  # <--- takes a long time

                ew1 = abs(ew1)
                ew2 = abs(ew2)

                conds[mode] = ew1.max() / ew2.min()
            else:
                conds[mode] = np.linalg.cond(A.todense())

    end = time.perf_counter()
    print(f" Time elapsed: {end - start: .5f} s")

    return V.ndof, conds
