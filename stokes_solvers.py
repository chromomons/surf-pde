# ------------------------------ LOAD LIBRARIES -------------------------------
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from xfem.lsetcurv import *
from ngsolve import solvers
from ngsolve import TaskManager
import time
from math import pi
import numpy as np


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
def errors(mesh, ds, Pmat, gfu, gfp, uSol, pSol):
    return sqrt(Integrate(InnerProduct(gfu - uSol, Pmat * (gfu - uSol)) * ds, mesh)),\
           sqrt(Integrate(InnerProduct((grad(gfu) - vec_grad(uSol)) * Pmat, Pmat * (grad(gfu) - vec_grad(uSol)) * Pmat) * ds, mesh)),\
           sqrt(Integrate((pSol - gfp) * (pSol - gfp) * ds, mesh)),\
           sqrt(Integrate(InnerProduct(grad(gfp) - coef_fun_grad(pSol), Pmat*(grad(gfp) - coef_fun_grad(pSol))) * ds, mesh))


def errors_ex(mesh, dX, n, gfu, gfp, ds, Pmat, Hmat, h, dt, rho_u, rho_p):
    errs = {
                'ngu': sqrt(Integrate(rho_u * InnerProduct(grad(gfu) * n, grad(gfu) * n) * dX, mesh)),
                'ngp': sqrt(Integrate(rho_p * InnerProduct(grad(gfp) * n, grad(gfp) * n) * dX, mesh)),
                'divg': sqrt(Integrate(InnerProduct(Trace(Pmat * grad(gfu) * Pmat) - (gfu*n)*Trace(Hmat), Trace(Pmat * grad(gfu) * Pmat) - (gfu*n)*Trace(Hmat)) * ds, mesh)),
                'tang': sqrt(Integrate((gfu*n) * (gfu*n) * ds, mesh))
            }
    return errs


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


# HELPERS
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

    return phi_kp1, n_k, phi_k, n_km1, Hmat


def define_forms(eq_type, V, Q, n, Pmat, Hmat, n_k, rhsf, rhsg, ds, dX, **args):
    u, v = V.TnT()
    p, q = Q.TnT()
    h = specialcf.mesh_size

    if eq_type == 'steady_stokes':
        # penalization parameters
        tau = 1.0 / (h * h)
        rho_u = 1.0 / h
        rho_p = 1.0 * h
        alpha = args['alpha']

        # A_h part
        a = BilinearForm(V, symmetric=True)
        a += (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                           Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        a += alpha * (InnerProduct(Pmat * u, Pmat * v)) * ds
        # penalization of the normal component of the velocity
        a += (tau * ((u * n_k) * (v * n_k))) * ds
        # normal gradient volume stabilization of the velocity
        a += (rho_u * InnerProduct(grad(u) * n, grad(v) * n)) * dX

        # b_T part
        b = BilinearForm(trialspace=V, testspace=Q)
        b += (InnerProduct(u, Pmat * grad(q))) * ds
        # normal gradient volume stabilization of the pressure,
        # c part
        c = BilinearForm(Q, symmetric=True)
        c += rho_p * ((grad(p) * n) * (grad(q) * n)) * dX

        # pressure preconditioner
        sq = BilinearForm(Q, symmetric=True)
        sq += p * q * ds
        sq += rho_p * ((grad(p) * n) * (grad(q) * n)) * dX

        f = LinearForm(V)
        f += rhsf * v * ds
        g = LinearForm(Q)
        g += (-1.0) * rhsg * q * ds

        return a, b, c, sq, f, g
    elif eq_type == 'stokes':
        # penalization parameters
        tau = 1.0 / (h * h)
        rho_u = 1.0 / h
        rho_p = 1.0 * h

        alpha = args['alpha']
        dt = args['dt']

        # Mass matrix
        m = BilinearForm(V, symmetric=True)
        m += InnerProduct(Pmat * u, Pmat * v) * ds

        # A_h part
        a = BilinearForm(V, symmetric=True)
        a += 2.0 / (alpha * dt) * InnerProduct(u, Pmat * v) * ds

        a += (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                           Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        # penalization of the normal component of the velocity
        a += (tau * ((u * n_k) * (v * n_k))) * ds
        a += (rho_u * InnerProduct(grad(u) * n, grad(v) * n)) * dX

        a2 = BilinearForm(V, symmetric=True)
        a2 += (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                            Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        # penalization of the normal component of the velocity
        a2 += (tau * ((u * n_k) * (v * n_k))) * ds
        # normal gradient volume stabilization of the velocity
        a2 += (rho_u * InnerProduct(grad(u) * n, grad(v) * n)) * dX

        # b_T part
        b = BilinearForm(trialspace=V, testspace=Q)
        b += (InnerProduct(u, Pmat * grad(q))) * ds
        # normal gradient volume stabilization of the pressure,
        # c part
        c = BilinearForm(Q, symmetric=True)
        c += rho_p * ((grad(p) * n) * (grad(q) * n)) * dX

        # pressure preconditioner
        sq = BilinearForm(Q, symmetric=True)
        sq += p * q * ds
        sq += rho_p * ((grad(p) * n) * (grad(q) * n)) * dX

        zerou = BilinearForm(V, symmetric=True)
        zerou += 0.0 * u * v * ds

        zeroq = BilinearForm(Q, symmetric=True)
        zeroq += 0.0 * p * q * ds

        f = LinearForm(V)
        f += rhsf * (Pmat * v) * ds
        g = LinearForm(Q)
        g += (-1.0) * rhsg * q * ds

        return m, a, a2, b, c, sq, zerou, zeroq, f, g
    else:
        # navier_stokes
        tau = 1.0 / (h * h)
        rho_u = 1.0 / h
        rho_p = 1.0 * h

        nu = args['nu']
        dt = args['dt']
        dtparam = args['dtparam']
        gfu_approx = args['gfu_approx']

        # velocity mass matrix
        m = BilinearForm(V, symmetric=True)
        m += InnerProduct(Pmat * u, Pmat * v) * ds

        # velocity mass-convection-total_stab_tests_diffusion matrix
        a = BilinearForm(V, symmetric=False)
        # mass part
        a += dtparam / dt * InnerProduct(Pmat * u, Pmat * v) * ds
        # total_stab_tests_diffusion part
        a += nu * (
            InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat, Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        # convection part
        a += InnerProduct((Pmat * grad(u) * Pmat - (u * n) * Hmat) * (Pmat * gfu_approx), Pmat * v) * ds
        # penalization of the normal component of the velocity
        a += (tau * ((u * n_k) * (v * n_k))) * ds
        # normal gradient in the bulk stabilization
        a += (rho_u * InnerProduct(grad(u) * n, grad(v) * n)) * dX

        # pressure mass-convection-total_stab_tests_diffusion matrix
        ap = BilinearForm(Q, symmetric=False)
        # mass part
        ap += dtparam / dt * p * q * ds
        # total_stab_tests_diffusion
        ap += nu * InnerProduct(Pmat * grad(p), Pmat * grad(q)) * ds
        # convection
        ap += InnerProduct(Pmat * grad(p), Pmat * gfu_approx) * q * ds
        # normal gradient in the bulk stabilization
        # SHOULD IT BE rho_p OR rho_u?
        ap += rho_p * ((grad(p) * n) * (grad(q) * n)) * dX
        # ap += (rho_p + rho_u) * ((grad(p) * n) * (grad(q) * n)) * dX

        # pressure total_stab_tests_diffusion matrix
        pd = BilinearForm(Q, symmetric=True)
        # total_stab_tests_diffusion
        pd += InnerProduct(Pmat * grad(p), Pmat * grad(q)) * ds
        # normal gradient in the bulk stabilization
        pd += rho_p * ((grad(p) * n) * (grad(q) * n)) * dX
        # pd += rho_u * ((grad(p) * n) * (grad(q) * n)) * dX

        # discrete divergence operator
        b = BilinearForm(trialspace=V, testspace=Q)
        b += (InnerProduct(u, Pmat * grad(q))) * ds

        # pressure stabilization matrix
        c = BilinearForm(Q, symmetric=True)
        c += rho_p * ((grad(p) * n) * (grad(q) * n)) * dX

        # stabilized pressure mass matrix
        sq = BilinearForm(Q, symmetric=True)
        sq += p * q * ds
        sq += rho_p * ((grad(p) * n) * (grad(q) * n)) * dX

        f = LinearForm(V)
        f += InnerProduct(rhsf, Pmat * v) * ds
        g = LinearForm(Q)
        g += (-1.0) * rhsg * q * ds

        return m, a, ap, pd, b, c, sq, f, g


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


def steady_stokes(mesh, alpha=1.0, order=2, out=False, **exact):
    phi = CoefficientFunction(exact["phi"]).Compile()
    ### Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(phi)
    lset_approx = lsetmeshadap.lset_p1
    mesh.SetDeformation(deformation)

    # FESpace: Taylor-Hood element
    VPhk   = VectorH1(mesh, order=order, dirichlet=[])
    Phkm1 = H1(mesh, order=order-1, dirichlet=[])

    ci = CutInfo(mesh, lset_approx)

    V = Compress(VPhk, GetDofsOfElements(VPhk, ci.GetElementsOfType(IF)))
    Q = Compress(Phkm1, GetDofsOfElements(Phkm1, ci.GetElementsOfType(IF)))

    # declare grid functions to store the solution
    gfu = GridFunction(V)
    gfp = GridFunction(Q)

    # helper grid functions
    phi_kp1, n_k, phi_k, n_km1, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=phi, vel_space=VPhk)

    # declare integration domains
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ci.GetElementsOfType(IF), deformation=deformation)
    dX = dx(definedonelements=ci.GetElementsOfType(IF), deformation=deformation)

    # unpacking exact solution
    uSol = CoefficientFunction((exact["u1"], exact["u2"], exact["u3"])).Compile()
    pSol = CoefficientFunction(exact["p"]).Compile()
    rhsf = CoefficientFunction((exact["f1"], exact["f2"], exact["f3"])).Compile()
    rhsg = CoefficientFunction(exact["g"]).Compile()

    # define projection matrix
    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)

    # bilinear forms:
    bilinear_form_args = {'alpha': alpha}
    a, b, c, sq, f, g = define_forms(eq_type='steady_stokes',
                                     V=V, Q=Q,
                                     n=n, Pmat=Pmat, Hmat=Hmat, n_k=n_k,
                                     rhsf=rhsf, rhsg=rhsg,
                                     ds=ds, dX=dX, **bilinear_form_args)

    start = time.perf_counter()
    with TaskManager():
        prea = Preconditioner(a, "bddc")
        presq = Preconditioner(sq, "bddc")

        assemble_forms([a, b, c, sq, f, g])

    print(f"{bcolors.OKCYAN}System assembled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    ### LINEAR SOLVER

    K = BlockMatrix([[a.mat, b.mat.T],
                     [b.mat, -c.mat]])

    inva = CGSolver(a.mat, prea.mat, maxsteps=5, precision=1e-6)
    invsq = CGSolver(sq.mat, presq.mat, maxsteps=5, precision=1e-6)

    C = BlockMatrix([[inva, None], [None, invsq]])

    rhs = BlockVector([f.vec, g.vec])
    sol = BlockVector([gfu.vec, gfp.vec])

    start = time.perf_counter()
    with TaskManager():
        solvers.MinRes(mat=K, pre=C, rhs=rhs, sol=sol, initialize=True, maxsteps=40, tol=1e-12)
    print(f"{bcolors.OKBLUE}System solved    ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    ### POST-PROCESSING

    # making numerical pressure mean zero
    with TaskManager():
        renormalize(Q, mesh, ds, gfp)

    ### ERRORS

    with TaskManager():
        l2u, h1u, l2p, h1p = errors(mesh, ds, Pmat, gfu, gfp, uSol, pSol)

    mesh.UnsetDeformation()

    if out:
        with TaskManager():
            vtk = VTKOutput(ma=mesh,
                            coefs=[lset_approx, deformation, gfu, gfp, uSol, pSol],
                            names=["P1-levelset", "deform", "u", "p", "uSol", "pSol"],
                            filename=f"steady-stokes-{exact['name']}", subdivision=0)
            vtk.Do()

    return V.ndof + Q.ndof, l2u, h1u, l2p, h1p


def stokes(mesh, dt, tfinal=1.0, order=2, out=False, **exact):
    phi = CoefficientFunction(exact["phi"]).Compile()
    # Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(phi)
    lset_approx = lsetmeshadap.lset_p1
    mesh.SetDeformation(deformation)

    alpha = 2.0 - np.sqrt(2.0)

    t = Parameter(0.0)
    tfun = 2. + sin(pi*t)
    tfun_dif = tfun.Diff(t)

    # FESpace: Taylor-Hood element
    VPhk   = VectorH1(mesh, order=order, dirichlet=[])
    Phkm1 = H1(mesh, order=order-1, dirichlet=[])

    ci = CutInfo(mesh, lset_approx)

    V = Compress(VPhk, GetDofsOfElements(VPhk, ci.GetElementsOfType(IF)))
    Q = Compress(Phkm1, GetDofsOfElements(Phkm1, ci.GetElementsOfType(IF)))

    # declare grid functions to store the solution
    gfu = GridFunction(V)
    gfp = GridFunction(Q)

    # helper grid functions
    phi_kp1, n_k, phi_k, n_km1, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=phi, vel_space=VPhk)

    # declare the integration domains
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ci.GetElementsOfType(IF), deformation=deformation)
    dX = dx(definedonelements=ci.GetElementsOfType(IF), deformation=deformation)

    domMeas = get_dom_measure(Q, mesh, ds)

    uSol = CoefficientFunction((tfun*exact["u1"], tfun*exact["u2"], tfun*exact["u3"])).Compile()
    pSol = CoefficientFunction(tfun*exact["p"]).Compile()
    rhsf = CoefficientFunction((tfun*exact["f1"] + tfun_dif*exact["u1"],
                                tfun*exact["f2"] + tfun_dif*exact["u2"],
                                tfun*exact["f3"] + tfun_dif*exact["u3"])).Compile()
    rhsg = CoefficientFunction(tfun*exact["g"]).Compile()

    # define projection matrix
    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)

    ### bilinear forms:
    bilinear_form_args = {'alpha': alpha, 'dt': dt}
    m, a, a2, b, c, sq, zerou, zeroq, f, g = define_forms(
        eq_type='stokes',
        V=V, Q=Q,
        n=n, Pmat=Pmat, Hmat=Hmat, n_k=n_k,
        rhsf=rhsf, rhsg=rhsg,
        ds=ds, dX=dX, **bilinear_form_args)

    start = time.perf_counter()
    with TaskManager():
        presq = Preconditioner(sq, "bddc")
        prea = Preconditioner(a, "bddc")

        assemble_forms([m, a, a2, b, c, sq, zerou, zeroq, f, g])

    print(f"{bcolors.OKCYAN}System assembled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    ### LINEAR SOLVER

    A = BlockMatrix([[a.mat, b.mat.T],
                     [b.mat, - c.mat]])
    Abdf2 = BlockMatrix([[a.mat, b.mat.T],
                         [b.mat, -((1.0 - alpha) / alpha) * c.mat]])
    A2bdf2 = BlockMatrix([[a2.mat, b.mat.T],
                          [b.mat, -((1.0 - alpha) / alpha) * c.mat]])
    Atr = BlockMatrix([[2.0*a2.mat, 2.0*b.mat.T],
                       [b.mat, -c.mat]])

    M = BlockMatrix([[m.mat, None],
                     [None, zeroq.mat]])

    maxsteps_minres = 40

    inva = CGSolver(a.mat, prea.mat, maxsteps=5, precision=1e-4)
    invsq = CGSolver(sq.mat, presq.mat, maxsteps=5, precision=1e-4)

    C = BlockMatrix([[inva, None],
                     [None, invsq]])

    U = BlockVector([gfu.vec, gfp.vec])
    diff = BlockVector([gfu.vec.CreateVector(), gfp.vec.CreateVector()])

    F = BlockVector([f.vec, g.vec])
    fold = f.vec.CreateVector()
    zerofunQ = GridFunction(Q)
    zerofunQ.Set(CoefficientFunction(0.0))
    Fold = BlockVector([fold, zerofunQ.vec])

    # TIME MARCHING

    t_curr = 0.0  # time counter within one block-run

    # IC
    t.Set(0.0)

    gfu.Set(uSol)
    gfp.Set(pSol)

    if out:
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, deformation, gfu, gfp, uSol, pSol],
                        names=["P1-levelset", "deform", "u", "p", "uSol", "pSol"],
                        filename=f"./vtk_out/parab/{exact['name']}",
                        subdivision=0)
        vtk.Do(time=0.0)

    rhs1 = f.vec.CreateVector()
    rhs2 = g.vec.CreateVector()
    rhs = BlockVector([rhs1, rhs2])

    out_errs = {'ts': [], 'l2us': [], 'h1us': [], 'l2ps': [], 'h1ps': []}

    with TaskManager():
        l2u, h1u, l2p, h1p = errors(mesh, ds, Pmat, gfu, gfp, uSol, pSol)
    append_errors(t_curr, l2u, h1u, l2p, h1p, **out_errs)

    start = time.perf_counter()

    fold.data = f.vec
    i = 0
    while t_curr < tfinal - 0.5 * dt:
        t.Set(t_curr + alpha*dt)
        with TaskManager():
            f.Assemble()
            g.Assemble()

        # TR
        rhs.data = Fold + F - Atr * U

        with TaskManager():
            solvers.MinRes(mat=A, pre=C, rhs=rhs, sol=diff, initialize=True, maxsteps=maxsteps_minres, tol=1e-12,
                           printrates=False)
            U.data += diff
            renormalize(Q, mesh, ds, gfp, domMeas)

        # BDF2
        t.Set(t_curr + dt)
        with TaskManager():
            f.Assemble()
            g.Assemble()

        rhs.data = F + (1.0-alpha)/(alpha*dt)*M*diff - A2bdf2 * U

        with TaskManager():
            solvers.MinRes(mat=Abdf2, pre=C, rhs=rhs, sol=diff, initialize=True, maxsteps=maxsteps_minres, tol=1e-12,
                           printrates=False)
            U.data += diff
            renormalize(Q, mesh, ds, gfp, domMeas)

        t_curr += dt

        fold.data = f.vec

        print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        with TaskManager():
            l2u, h1u, l2p, h1p = errors(mesh, ds, Pmat, gfu, gfp, uSol, pSol)
        append_errors(t_curr, l2u, h1u, l2p, h1p, **out_errs)

        if out:
            with TaskManager():
                vtk.Do(time=t_curr)
        i += 1

    print("")
    end = time.perf_counter()
    print(f" Time elapsed: {end - start: .5f} s")

    mesh.UnsetDeformation()

    return V.ndof + Q.ndof, out_errs['ts'], out_errs['l2us'], out_errs['h1us'], out_errs['l2ps'], out_errs['h1ps']


def navier_stokes(mesh, dt, tfinal=1.0, order=2, out=False, printrates=False, outer_gmres_iter=20, nref=0, tol=1e-12, n_gmres = 50, is_reltol=True, **exact):
    nu = exact['nu']
    phi = CoefficientFunction(exact["phi"]).Compile()
    # Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(phi)
    lset_approx = lsetmeshadap.lset_p1
    mesh.SetDeformation(deformation)

    t = Parameter(-dt)
    tfun = 1.0 + sin(pi * t)
    tfun_dif = tfun.Diff(t)

    # FESpace: Taylor-Hood element
    VPhk = VectorH1(mesh, order=order, dirichlet=[])
    Phkm1 = H1(mesh, order=order - 1, dirichlet=[])

    ci = CutInfo(mesh, lset_approx)

    V = Compress(VPhk, GetDofsOfElements(VPhk, ci.GetElementsOfType(IF)))
    Q = Compress(Phkm1, GetDofsOfElements(Phkm1, ci.GetElementsOfType(IF)))

    # declare grid functions to store the solution
    gfu = GridFunction(V)
    gfu_prev = GridFunction(V)
    gfu_approx = GridFunction(V)
    gfp = GridFunction(Q)

    # helper grid functions
    phi_kp1, n_k, phi_k, n_km1, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=phi, vel_space=VPhk)

    # declare the integration domains
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ci.GetElementsOfType(IF), deformation=deformation)
    dX = dx(definedonelements=ci.GetElementsOfType(IF), deformation=deformation)

    domMeas = get_dom_measure(Q, mesh, ds)

    uSol = CoefficientFunction((tfun * exact["u1"], tfun * exact["u2"], tfun * exact["u3"])).Compile()
    pSol = CoefficientFunction(tfun * exact["p"]).Compile()
    rhsf = CoefficientFunction((tfun * exact["f1"] + tfun_dif * exact["u1"] + tfun*tfun*exact["conv1"],
                                tfun * exact["f2"] + tfun_dif * exact["u2"] + tfun*tfun*exact["conv2"],
                                tfun * exact["f3"] + tfun_dif * exact["u3"] + tfun*tfun*exact["conv3"])).Compile()
    rhsg = CoefficientFunction(tfun * exact["g"]).Compile()

    # define projection matrix
    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)

    t.Set(-dt)
    gfu_prev.Set(uSol)

    t.Set(0.0)
    gfu.Set(uSol)
    gfp.Set(pSol)

    gfu_approx.Set(2. * gfu - gfu_prev)
    dtparam = 3./2

    # BI-LINEAR FORMS
    bilinear_form_args = {'nu': nu, 'dt': dt, 'dtparam': dtparam, 'gfu_approx': gfu_approx}

    m, a, ap, pd, b, c, sq, f, g = define_forms(
        eq_type='ns',
        V=V, Q=Q,
        n=n, Pmat=Pmat, Hmat=Hmat, n_k=n_k,
        rhsf=rhsf, rhsg=rhsg,
        ds=ds, dX=dX, **bilinear_form_args
    )

    start = time.perf_counter()
    with TaskManager():
        presq = Preconditioner(sq, "bddc")
        prepd = Preconditioner(pd, "bddc")
        prea = Preconditioner(a, "local")

        assemble_forms([b, c, sq, m, pd])

    print(f"{bcolors.OKCYAN}System assembled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    # LINEAR SOLVER

    invsq = CGSolver(sq.mat, presq.mat, maxsteps=5, precision=1e-6)
    invpd = CGSolver(pd.mat, prepd.mat, maxsteps=5, precision=1e-6)

    U = BlockVector([gfu.vec, gfp.vec])

    # TIME MARCHING

    t_curr = 0.0  # time counter within one block-run

    if out:
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, deformation, gfu, gfp, uSol, pSol],
                        names=["P1-levelset", "deform", "u", "p", "uSol", "pSol"],
                        filename=f"./vtk_out/parab/ns-{exact['name']}-{nu:.2f}",
                        subdivision=0)
        vtk.Do(time=0.0)

    rhs = BlockVector([f.vec.CreateVector(),
                       g.vec.CreateVector()])

    diffu = 0.0 * gfu.vec.CreateVector()
    diffp = 0.0 * gfp.vec.CreateVector()
    diff = BlockVector([diffu, diffp])

    out_errs = {'ts': [], 'l2us': [], 'h1us': [], 'l2ps': [], 'h1ps': []}

    with TaskManager():
        l2u, h1u, l2p, h1p = errors(mesh, ds, Pmat, gfu, gfp, uSol, pSol)
    append_errors(t_curr, l2u, h1u, l2p, h1p, **out_errs)

    start = time.perf_counter()

    while t_curr < tfinal - 0.5 * dt:
        t.Set(t_curr + dt)
        gfu_approx.Set(2.0 * gfu - gfu_prev)
        with TaskManager():
            assemble_forms([f, g, a, ap])

            inva = GMRESSolver(a.mat, prea.mat, maxsteps=n_gmres*int(2**(nref)), precision=1e-6)
            invms = invsq @ ap.mat @ invpd

            A = BlockMatrix([[a.mat, b.mat.T],
                             [b.mat, -c.mat]])

            C = BlockMatrix([[inva, inva @ b.mat.T @ invms],
                             [None, -invms]])

            F = BlockVector([f.vec + m.mat * (2.0/dt * gfu.vec - 0.5/dt * gfu_prev.vec),
                             g.vec])

            rhs.data = F - A * U

            gfu_prev.Set(gfu)
            diff = BlockVector([0.0 * diffu, 0.0 * diffp])

            if is_reltol:
                solvers.GMRes(A=A, b=rhs, pre=C, x=diff, printrates=printrates, maxsteps=outer_gmres_iter, reltol=tol)
            else:
                solvers.GMRes(A=A, b=rhs, pre=C, x=diff, printrates=printrates, maxsteps=outer_gmres_iter, tol=tol)
            U += diff

            renormalize(Q, mesh, ds, gfp, domMeas)

        t_curr += dt

        if printrates:
            print(f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)")
        else:
            print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        with TaskManager():
            l2u, h1u, l2p, h1p = errors(mesh, ds, Pmat, gfu, gfp, uSol, pSol)
        append_errors(t_curr, l2u, h1u, l2p, h1p, **out_errs)

        if out:
            with TaskManager():
                vtk.Do(time=t_curr)

    print("")
    end = time.perf_counter()
    print(f" Time elapsed: {end - start: .5f} s")

    mesh.UnsetDeformation()

    return V.ndof + Q.ndof, out_errs['ts'], out_errs['l2us'], out_errs['h1us'], out_errs['l2ps'], out_errs['h1ps']
