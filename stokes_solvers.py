# ------------------------------ LOAD LIBRARIES -------------------------------
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
from xfem.lsetcurv import *
from ngsolve import solvers
from ngsolve import TaskManager
import time
import numpy as np
from utils import bcolors, assemble_forms, get_dom_measure, renormalize, mass_append, errors_scal, \
    errors_vec, helper_grid_functions, update_geometry


# EXACT SOLUTION CLASS
# HELPERS

def set_ic(mesh, V, order, gfu_prevs, exact, dt,
           lsetmeshadap, lset_approx, band, ba_IF, ba_IF_band,
           n, Pmat, rho_u, tau, ds, dX, precond_name, cg_iter):
    """
    A routine that sets initial condition for BDFk method with normal extension on a narrowband around the surface.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma and is (ideally) refined around it.
        V: VectorH1
            Ambient FE space for velocity.
        order: int
            Polynomial order of the velocity approximation.
        gfu_prevs: List[GridFunction(V)]
            List of GridFunction's defined on the ambient velocity space. Should contain k grid function for the BDFk
            method.
        exact: Exact
            Exact solution object, see stokes_solvers.py file.
        dt: float
            Time step size.
        lsetmeshadap: xfem.lsetcurv.LevelSetMeshAdaption
            An object used to achieve higher-order approximation of the geometry via deformation.
        lset_approx: P1 GridFunction on mesh.
            Variable for P1 approximation of the levelset.
        band: float
            Size of the band around levelset, the distance metric is in terms of the levelset function.
        ba_IF: BitArray
            Stores element numbers that are intersected by the surface.
        ba_IF_band: BitArray
            Stores element numbers that are in the narrowband around the surface.
        n: Vector-valued GridFunction
            Discrete normal vector
        Pmat: Tensor-valued GridFunction
            Discrete projection matrix
        rho_u: GridFunction
            Normal gradient stabilization parameter for the velocity.
        tau: GridFunction
            Parameter for the penalization of the normal component of the velocity field.
        ds: xfem.dCul
            Element of area on Gamma.
        dX: ngsolve.utils.dx
            Element of bulk volume in the band around surface.
        precond_name: str
            Name of the preconditioner, either "bddc" or "local" (Jacobi)
        cg_iter: int
            Number of CG iterations.

    Returns:

    """
    time_order = len(gfu_prevs)

    coef_phi = CoefficientFunction(exact.cfs["phi"]).Compile()
    coef_fel = CoefficientFunction((exact.cfs["fel1"], exact.cfs["fel2"], exact.cfs["fel3"])).Compile()
    for j in range(time_order):
        # fix levelset
        exact.set_time(-j * dt)

        deformation = lsetmeshadap.CalcDeformation(coef_phi)

        # solve elliptic problem on a fixed surface to get u with normal extension
        update_geometry(mesh, coef_phi, lset_approx, band, ba_IF, ba_IF_band)
        VG = Compress(V, GetDofsOfElements(V, ba_IF_band))

        # helper grid functions
        n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=coef_phi, vel_space=V)

        gfu_el = GridFunction(VG)

        u_el, v_el = VG.TnT()

        a_el = BilinearForm(VG, symmetric=True)
        a_el += InnerProduct(Pmat * u_el, Pmat * v_el) * ds
        a_el += (InnerProduct(Pmat * Sym(grad(u_el)) * Pmat - (u_el * n) * Hmat,
                              Pmat * Sym(grad(v_el)) * Pmat - (v_el * n) * Hmat)) * ds
        a_el += (tau * ((u_el * n_k) * (v_el * n_k))) * ds
        a_el += (rho_u * InnerProduct(grad(u_el) * n, grad(v_el) * n)) * dX

        f_el = LinearForm(VG)
        f_el += InnerProduct(coef_fel, Pmat * v_el) * ds

        with TaskManager():
            pre_a_el = Preconditioner(a_el, precond_name)

            a_el.Assemble()
            f_el.Assemble()

            solvers.CG(mat=a_el.mat, rhs=f_el.vec, pre=pre_a_el.mat, sol=gfu_el.vec, printrates=False, maxsteps=cg_iter)
            sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

            gfu_prevs[j].Set(gfu_el)


def define_forms_moving_ns(
        VG, QG,
        bdf_coeff, dt,
        n, Pmat, n_k, Hmat,
        tau, rho_u, rho_p,
        ds, dX, dX2,
        param_rho, param_mu, coef_wN, coef_f, coef_g,
        gfu_approx, gfu_prevs,
        solver='gmres', XG=None):
    """
    Routine that defines bilinear and linear forms for the evolving-surface Navier-Stokes for GMRes and direct solvers.
    Args:
        VG: Compressed VectorH1 space
            TraceFEM space for u.
        QG: Compressed H1 space
            TraceFEM space for p.
        bdf_coeff: List[float]
            List of BDF coefficients.
        dt: float
            Time step size.
        n: Vector-valued GridFunction
            Discrete normal vector
        Pmat: Tensor-valued GridFunction
            Discrete projection matrix
        n_k: Vector-valued GridFunction
            Higher-order approximation of the normal vector
        Hmat: Tensor-valued GridFunction
            Discrete shape operator
        tau: GridFunction
            Parameter for the penalization of the normal component of the velocity field.
        rho_u: GridFunction
            Normal gradient stabilization parameter for the velocity.
        rho_p: GridFunction
            Normal gradient stabilization parameter for the pressure.
        ds: xfem.dCul
            Element of area on Gamma.
        dX: ngsolve.utils.dx
            Element of bulk volume in the band around surface.
        dX2: ngsolve.utils.dx
            Element of bulk volume elements cut by surface.
        coef_wN: CoefficientFunction
            Normal component of the ambient velocity field (scalar quantity).
        coef_f: Vector-valued CoefficientFunction
            Right-hand side of the momentum equation.
        coef_g: CoefficientFunction
            Right-hand side of the continuity equation.
        gfu_approx: Vector-valued GridFunction
            Extrapolation of u at t=t_{n+1} for linearization of the convection term in Navier-Stokes
        gfu_prevs: List[GridFunction(V)]
            List of GridFunction's defined on the ambient velocity space. Should contain k grid function for the BDFk
            method.
        solver: str
            Solver used: "gmres" or "direct" (default).
        XG: Compressed VectorH1 x H1 space
            Compressed FE space for velocity-pressure pair. Needed for direct solver.

    Returns:

    """
    time_order = len(bdf_coeff)-1

    if solver == 'gmres':
        u, v = VG.TnT()
        p, q = QG.TnT()

        a = BilinearForm(VG, symmetric=False)
        a += param_rho * bdf_coeff[0] / dt * InnerProduct(u, Pmat * v) * ds
        a += param_rho * coef_wN * InnerProduct(Hmat * u, Pmat * v) * ds
        a += param_rho * 0.5 * InnerProduct((Pmat * grad(u) * Pmat - (u * n) * Hmat) * gfu_approx, v) * ds
        a += param_rho * (-0.5) * InnerProduct((Pmat * grad(v) * Pmat - (v * n) * Hmat) * gfu_approx, u) * ds
        a += param_rho * (-0.5) * InnerProduct(coef_g * u, Pmat * v) * ds
        a += 2.0 * param_mu * (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                                            Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        a += param_rho * tau * InnerProduct(n_k, u) * InnerProduct(n_k, v) * ds
        a += 2.0 * param_mu * rho_u * InnerProduct(grad(u) * n, grad(v) * n) * dX

        # pressure mass-convection-diffusion matrix
        ap = BilinearForm(QG, symmetric=False)
        # mass part
        ap += param_rho * bdf_coeff[0] / dt * p * q * ds
        # convection
        ap += param_rho * InnerProduct(Pmat * grad(p), Pmat * gfu_approx) * q * ds
        # diffusion part
        ap += 2 * param_mu * InnerProduct(Pmat * grad(p), Pmat * grad(q)) * ds
        # normal gradient in the bulk stabilization
        # SHOULD IT BE rho_p OR rho_u?
        ap += 2 * param_mu * rho_p * ((grad(p) * n) * (grad(q) * n)) * dX2

        # pressure diffusion matrix
        pd = BilinearForm(QG, symmetric=True)
        # diffusion
        pd += 2 * param_mu * InnerProduct(Pmat * grad(p), Pmat * grad(q)) * ds
        # normal gradient in the bulk stabilization
        pd += 2 * param_mu * rho_p * ((grad(p) * n) * (grad(q) * n)) * dX2

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
            f += (-1.0) * bdf_coeff[j + 1] / dt * InnerProduct(gfu_prevs[j], Pmat * v) * ds

        g = LinearForm(QG)
        g += (-1.0) * coef_g * q * ds

        return a, ap, pd, sq, b, c, f, g
    else:
        u, p = XG.TrialFunction()
        v, q = XG.TestFunction()

        a = BilinearForm(XG, symmetric=False)
        a += param_rho * bdf_coeff[0] / dt * InnerProduct(u, Pmat * v) * ds
        a += param_rho * coef_wN * InnerProduct(Hmat * u, Pmat * v) * ds
        a += param_rho * 0.5 * InnerProduct((Pmat * grad(u) * Pmat - (u * n) * Hmat) * gfu_approx, v) * ds
        a += param_rho * (-0.5) * InnerProduct((Pmat * grad(v) * Pmat - (v * n) * Hmat) * gfu_approx, u) * ds
        a += param_rho * (-0.5) * InnerProduct(coef_g * u, Pmat * v) * ds
        a += 2.0 * param_mu * (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                                            Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        a += param_rho * tau * InnerProduct(n_k, u) * InnerProduct(n_k, v) * ds
        a += 2.0 * param_mu * rho_u * InnerProduct(grad(u) * n, grad(v) * n) * dX

        a += InnerProduct(u, Pmat * grad(q)) * ds
        a += InnerProduct(v, Pmat * grad(p)) * ds

        a += (-1.0) * rho_p * (grad(p) * n) * (grad(q) * n) * dX2

        f = LinearForm(XG)
        f += InnerProduct(coef_f, Pmat * v) * ds
        for j in range(time_order):
            f += (-1.0) * bdf_coeff[j + 1] / dt * InnerProduct(gfu_prevs[j], Pmat * v) * ds
        f += (-1.0) * coef_g * q * ds

        return a, f


def define_forms(eq_type, V, Q, n, Pmat, Hmat, n_k, coef_f, coef_g, ds, dX, **args):
    """
    Routine that defines bilinear and linear forms for the steady and unsteady Stokes, and unsteady Navier-Stokes.
    Args:
        eq_type: str
            Type of PDE: "steady_stokes", "stokes" or "navier-stokes" (default)
        V: Compressed VectorH1 space
            TraceFEM space for u (velocity)
        Q: Compressed H1 space
            TraceFEM space for p (pressure)
        n: Vector-valued GridFunction
            Discrete normal vector
        Pmat: Tensor-valued GridFunction
            Discrete projection matrix
        Hmat: Tensor-valued GridFunction
            Discrete shape operator
        n_k: Vector-valued GridFunction
            Higher-order approximation of the normal vector
        coef_f: Vector-valued CoefficientFunction
            Right-hand side of the momentum equation
        coef_g: CoefficientFunction
            Right-hand side of the continuity equation
        ds: xfem.dCul
            Element of area on Gamma
        dX: ngsolve.utils.dx
            Element of bulk volume
        **args:
        Parameters that are problem-dependent, like
            args['alpha']: float
                Coefficient in front of the mass term for steady Stokes, or parameter for TR-BDF2 for unsteady Stokes
                (will need to change this confusing notation)
            args['nu']: float
                Coefficient of kinematic viscocity
            args['dt']: float
                Time step size
            args['nu']: float
                Coefficient of kinematic viscocity for Navier-Stokes
            args['dt_param']: float
                Coefficient in front of the mass term for time discretizations (e.g. 3/2 for BDF2)
            args['gfu_approx']: Vector-valued GridFunction
                Extrapolation of u at t=t_{n+1} for linearization of the convection term in Navier-Stokes
    Returns:
        A tuple of bilinear and linear problems relevant to the problem
    """
    u, v = V.TnT()
    p, q = Q.TnT()
    h = specialcf.mesh_size

    if eq_type == 'steady_stokes':
        # penalization parameters
        tau = 1.0 / (h * h)
        rho_u = 1.0 / h
        rho_p = 1.0 * h
        param_alpha = args['param_alpha']
        param_mu = args['param_mu']

        # A_h part
        a = BilinearForm(V, symmetric=True)
        a += param_alpha * (InnerProduct(Pmat * u, Pmat * v)) * ds
        a += 2 * param_mu * (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                                          Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        # penalization of the normal component of the velocity
        a += param_alpha * (tau * ((u * n_k) * (v * n_k))) * ds
        # normal gradient volume stabilization of the velocity
        a += 2 * param_mu * (rho_u * InnerProduct(grad(u) * n, grad(v) * n)) * dX

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
        f += InnerProduct(coef_f, Pmat * v) * ds
        g = LinearForm(Q)
        g += (-1.0) * coef_g * q * ds

        return a, b, c, sq, f, g
    elif eq_type == 'stokes':
        # penalization parameters
        tau = 1.0 / (h * h)
        rho_u = 1.0 / h
        rho_p = 1.0 * h

        param_rho = args['param_rho']
        param_mu = args['param_mu']
        alpha = args['alpha']
        dt = args['dt']

        # Mass matrix
        m = BilinearForm(V, symmetric=True)
        m += param_rho * InnerProduct(Pmat * u, Pmat * v) * ds

        # A_h part
        a = BilinearForm(V, symmetric=True)
        a += param_rho * 2.0 / (alpha * dt) * InnerProduct(Pmat * u, Pmat * v) * ds

        a += 2 * param_mu * (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                                          Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        # penalization of the normal component of the velocity
        a += param_rho * (tau * ((u * n_k) * (v * n_k))) * ds
        a += 2 * param_mu * (rho_u * InnerProduct(grad(u) * n, grad(v) * n)) * dX

        a2 = BilinearForm(V, symmetric=True)
        a2 += 2 * param_mu * (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                                           Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        # penalization of the normal component of the velocity
        a2 += param_rho * (tau * ((u * n_k) * (v * n_k))) * ds
        # normal gradient volume stabilization of the velocity
        a2 += 2 * param_mu * (rho_u * InnerProduct(grad(u) * n, grad(v) * n)) * dX

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
        f += coef_f * (Pmat * v) * ds
        g = LinearForm(Q)
        g += (-1.0) * coef_g * q * ds

        return m, a, a2, b, c, sq, zerou, zeroq, f, g
    else:
        # navier_stokes
        tau = 1.0 / (h * h)
        rho_u = 1.0 / h
        rho_p = 1.0 * h

        param_rho = args['param_rho']
        param_mu = args['param_mu']
        dt = args['dt']
        bdf_coeff = args['bdf_coeff']
        gfu_approx = args['gfu_approx']

        # velocity mass matrix
        m = BilinearForm(V, symmetric=True)
        m += InnerProduct(Pmat * u, Pmat * v) * ds

        a = BilinearForm(V, symmetric=False)
        # mass part
        a += param_rho * bdf_coeff / dt * InnerProduct(Pmat * u, Pmat * v) * ds
        # diffusion part
        a += 2 * param_mu * (
            InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat, Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        # convection part, skew-symmetrized
        # a += InnerProduct((Pmat * grad(u) * Pmat - (u * n) * Hmat) * (Pmat * gfu_approx), Pmat * v) * ds
        a += 0.5 * param_rho * InnerProduct((Pmat * grad(u) * Pmat - (u * n) * Hmat) * (Pmat * gfu_approx), Pmat * v) * ds
        a += (-0.5 * param_rho) * InnerProduct((Pmat * grad(v) * Pmat - (v * n) * Hmat) * (Pmat * gfu_approx), Pmat * u) * ds
        a += (-0.5 * param_rho) * coef_g * InnerProduct(Pmat * u, Pmat * v) * ds
        # penalization of the normal component of the velocity
        a += param_rho * tau * ((u * n_k) * (v * n_k)) * ds
        # normal gradient in the bulk stabilization
        a += 2 * param_mu * (rho_u * InnerProduct(grad(u) * n, grad(v) * n)) * dX

        # pressure mass-convection-diffusion matrix
        ap = BilinearForm(Q, symmetric=False)
        # mass part
        ap += param_rho * bdf_coeff / dt * p * q * ds
        # convection
        ap += param_rho * InnerProduct(Pmat * grad(p), Pmat * gfu_approx) * q * ds
        # diffusion part
        ap += 2 * param_mu * InnerProduct(Pmat * grad(p), Pmat * grad(q)) * ds
        # normal gradient in the bulk stabilization
        ap += 2 * param_mu * rho_p * ((grad(p) * n) * (grad(q) * n)) * dX

        # pressure diffusion matrix
        pd = BilinearForm(Q, symmetric=True)
        # total_stab_tests_diffusion
        pd += 2 * param_mu * InnerProduct(Pmat * grad(p), Pmat * grad(q)) * ds
        # normal gradient in the bulk stabilization
        pd += 2 * param_mu * rho_p * ((grad(p) * n) * (grad(q) * n)) * dX

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
        f += InnerProduct(coef_f, Pmat * v) * ds
        g = LinearForm(Q)
        g += (-1.0) * coef_g * q * ds

        return m, a, ap, pd, b, c, sq, f, g


# SOLVERS
def steady_stokes(mesh, exact, order, linear_solver_params, vtk_out=None, logs=True, printrates=False):
    """
    Solves Steady Stokes problem (with an added mass term to allow non-mean-zero right-hand-sides) on a provided mesh
    using P_k-P_{k-1} Taylor-Hood pair. The initial data and RHS needs to be specified in a dictionary exact. VTK output
    can be provided if enabled.
    See http://arxiv.org/abs/1801.06589, http://arxiv.org/abs/2103.03843 for details of the scheme.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma and is (ideally) refined around it.
        exact: exact.Exact
            Exact solution object, see exact.py and fixed_surface_poisson.py
        order: int
            Polynomial order for FEM for velocity.
        linear_solver_params: dict
            A dictionary with number of iterations for linear solvers.
        vtk_out: str
            String to be appended to the name of the VTK file.
        logs: bool
            Flag that indicates if logs are to be printed.
        printrates: bool
            Flag that indicates if linear solver residuals are to be printed.

    Returns:
        V.ndof + Q.ndof: int
            Number of degrees of freedom of the problem.
        l2u: float
            L^2-error of the velocity over Gamma
        h1u: float
            H^1-error of the velocity over Gamma
        l2p: float
            L^2-error of the pressure over Gamma
        h1p: float
            H^1-error of the pressure over Gamma
    """
    if order < 3:
        precond_name = "bddc"
        cg_iter = linear_solver_params['bddc_cg_iter']
    else:
        precond_name = "local"
        cg_iter = linear_solver_params['jacobi_cg_iter']
    minres_iter = linear_solver_params['minres_iter']

    # unpack exact
    param_mu = exact.params['mu']
    param_alpha = exact.params['alpha']

    coef_phi = CoefficientFunction(exact.cfs["phi"]).Compile()
    coef_u = CoefficientFunction((exact.cfs["u1"], exact.cfs["u2"], exact.cfs["u3"])).Compile()
    coef_p = CoefficientFunction(exact.cfs['p']).Compile()
    coef_f = CoefficientFunction((exact.cfs["f1"], exact.cfs["f2"], exact.cfs["f3"])).Compile()
    coef_g = CoefficientFunction(exact.cfs['g']).Compile()

    ### Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(coef_phi)
    lset_approx = lsetmeshadap.lset_p1

    # FESpace: Taylor-Hood element
    VPhk = VectorH1(mesh, order=order, dirichlet=[])
    Phkm1 = H1(mesh, order=order-1, dirichlet=[])

    ci = CutInfo(mesh, lset_approx)

    V = Compress(VPhk, GetDofsOfElements(VPhk, ci.GetElementsOfType(IF)))
    Q = Compress(Phkm1, GetDofsOfElements(Phkm1, ci.GetElementsOfType(IF)))

    # declare grid functions to store the solution
    gfu = GridFunction(V)
    gfp = GridFunction(Q)

    # helper grid functions
    n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=coef_phi, vel_space=VPhk)

    # declare integration domains
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ci.GetElementsOfType(IF), deformation=deformation)
    dX = dx(definedonelements=ci.GetElementsOfType(IF), deformation=deformation)

    # define projection matrix
    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)

    # bilinear forms:
    bilinear_form_args = {'param_alpha': param_alpha, 'param_mu': param_mu}
    a, b, c, sq, f, g = define_forms(
        eq_type='steady_stokes',
        V=V, Q=Q,
        n=n, Pmat=Pmat, Hmat=Hmat, n_k=n_k,
        coef_f=coef_f, coef_g=coef_g,
        ds=ds, dX=dX, **bilinear_form_args
    )

    start = time.perf_counter()
    with TaskManager():
        prea = Preconditioner(a, precond_name)
        presq = Preconditioner(sq, precond_name)

        assemble_forms([a, b, c, sq, f, g])

    if logs:
        print(f"{bcolors.OKCYAN}System assembled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    ### LINEAR SOLVER

    K = BlockMatrix([[a.mat, b.mat.T],
                     [b.mat, -c.mat]])

    inva = CGSolver(a.mat, prea.mat, maxsteps=cg_iter, precision=1e-4)
    invsq = CGSolver(sq.mat, presq.mat, maxsteps=cg_iter, precision=1e-4)

    C = BlockMatrix([[inva, None], [None, invsq]])

    rhs = BlockVector([f.vec, g.vec])
    sol = BlockVector([gfu.vec, gfp.vec])

    start = time.perf_counter()
    with TaskManager():
        solvers.MinRes(mat=K, pre=C, rhs=rhs, sol=sol, initialize=True, maxsteps=minres_iter, printrates=printrates,
                       tol=1e-12)
    if logs:
        print(f"{bcolors.OKBLUE}System solved    ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    ### POST-PROCESSING

    # making numerical pressure mean zero
    with TaskManager():
        renormalize(Q, mesh, ds, gfp)

    ### ERRORS

    with TaskManager():
        l2p, h1p = errors_scal(mesh, ds, Pmat, gfp, coef_p)
        l2u, h1u = errors_vec(mesh, ds, Pmat, gfu, coef_u)

    if vtk_out:
        with TaskManager():
            vtk = VTKOutput(ma=mesh,
                            coefs=[lset_approx, deformation, gfu, gfp, coef_u, coef_p],
                            names=["lset_p1", "deform", "u", "p", "coef_u", "coef_p"],
                            filename=f"./output/vtk_out/fixed_surface_steady_stokes_p{order}_{exact.name}_{vtk_out}",
                            subdivision=0)
            vtk.Do()

    return V.ndof + Q.ndof, l2u, h1u, l2p, h1p


def stokes(mesh, exact, dt, tfinal, order, linear_solver_params, vtk_out=None, logs=True, printrates=False):
    """
    Solves Unsteady Stokes problem on a provided mesh using P_k-P_{k-1} Taylor-Hood pair.
    The initial data and RHS needs to be specified in a dictionary exact. VTK output can be provided if enabled.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma and is (ideally) refined around it.
        exact: exact.Exact
            Exact solution object, see exact.py and fixed_surface_poisson.py
        dt: float
            Time step size.
        tfinal: float
            Final time in the simulation.
        order: int
            Polynomial order for FEM for velocity.
        linear_solver_params: dict
            A dictionary with number of iterations for linear solvers.
        vtk_out: str
            String to be appended to the name of the VTK file.
        logs: bool
            Flag that indicates if logs are to be printed.
        printrates: bool
            Flag that indicates if linear solver residuals are to be printed.

    Returns:
        V.ndof + Q.ndof: int
            Number of degrees of freedom of the problem.
        ts: List[float]
            Discrete times t_n at which problem was solved.
        l2us: List[float]
            L^2-error of the velocity over Gamma at each time t_n
        h1us: List[float]
            H^1-error of the velocity over Gamma at each time t_n
        l2ps: List[float]
            L^2-error of the pressure over Gamma at each time t_n
        h1ps: List[float]
            H^1-error of the pressure over Gamma at each time t_n
    """
    if order < 3:
        precond_name = "bddc"
        cg_iter = linear_solver_params['bddc_cg_iter']
    else:
        precond_name = "local"
        cg_iter = linear_solver_params['jacobi_cg_iter']
    minres_iter = linear_solver_params['minres_iter']

    # unpack exact
    param_rho = exact.params['rho']
    param_mu = exact.params['mu']

    coef_phi = CoefficientFunction(exact.cfs["phi"]).Compile()
    coef_u = CoefficientFunction((exact.cfs["u1"], exact.cfs["u2"], exact.cfs["u3"])).Compile()
    coef_p = CoefficientFunction(exact.cfs['p']).Compile()
    coef_f = CoefficientFunction((exact.cfs["f1"], exact.cfs["f2"], exact.cfs["f3"])).Compile()
    coef_g = CoefficientFunction(exact.cfs['g']).Compile()

    # Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(coef_phi)
    lset_approx = lsetmeshadap.lset_p1

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
    n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=coef_phi, vel_space=VPhk)

    # declare the integration domains
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ci.GetElementsOfType(IF), deformation=deformation)
    dX = dx(definedonelements=ci.GetElementsOfType(IF), deformation=deformation)

    domMeas = get_dom_measure(Q, mesh, ds)

    # define projection matrix
    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)

    # Parameter for TR-BDF2
    alpha = 2.0 - np.sqrt(2.0)

    ### bilinear forms:
    bilinear_form_args = {'param_rho': param_rho, 'param_mu': param_mu, 'alpha': alpha, 'dt': dt}
    m, a, a2, b, c, sq, zerou, zeroq, f, g = define_forms(
        eq_type='stokes',
        V=V, Q=Q,
        n=n, Pmat=Pmat, Hmat=Hmat, n_k=n_k,
        coef_f=coef_f, coef_g=coef_g,
        ds=ds, dX=dX, **bilinear_form_args
    )

    start = time.perf_counter()
    with TaskManager():
        prea = Preconditioner(a, precond_name)
        presq = Preconditioner(sq, precond_name)

        assemble_forms([m, a, a2, b, c, sq, zerou, zeroq, f, g])

    if logs:
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

    inva = CGSolver(a.mat, prea.mat, maxsteps=cg_iter, precision=1e-4)
    invsq = CGSolver(sq.mat, presq.mat, maxsteps=cg_iter, precision=1e-4)

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
    exact.set_time(0.0)

    mesh.SetDeformation(deformation)
    gfu.Set(coef_u)
    gfp.Set(coef_p)
    mesh.UnsetDeformation()

    if vtk_out:
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, deformation, gfu, gfp, coef_u, coef_p],
                        names=["lset_p1", "deform", "u", "p", "coef_u", "coef_p"],
                        filename=f"./output/vtk_out/fixed_surface_steady_stokes_p{order}_{exact.name}_{vtk_out}",
                        subdivision=0)
        vtk.Do(time=0.0)

    rhs1 = f.vec.CreateVector()
    rhs2 = g.vec.CreateVector()
    rhs = BlockVector([rhs1, rhs2])

    keys = ['ts', 'l2us', 'h1us', 'l2ps', 'h1ps']
    out_errs = {'ts': [], 'l2us': [], 'h1us': [], 'l2ps': [], 'h1ps': []}

    with TaskManager():
        l2p, h1p = errors_scal(mesh, ds, Pmat, gfp, coef_p)
        l2u, h1u = errors_vec(mesh, ds, Pmat, gfu, coef_u)
    mass_append(keys=keys, vals=[t_curr, l2u, h1u, l2p, h1p], **out_errs)

    start = time.perf_counter()

    fold.data = f.vec
    i = 0
    while t_curr < tfinal - 0.5 * dt:
        exact.set_time(t_curr + alpha*dt)
        with TaskManager():
            f.Assemble()
            g.Assemble()

        # TR
        rhs.data = Fold + F - Atr * U

        with TaskManager():
            solvers.MinRes(mat=A, pre=C, rhs=rhs, sol=diff, initialize=True, maxsteps=minres_iter, tol=1e-12,
                           printrates=printrates)
            U.data += diff
            renormalize(Q, mesh, ds, gfp, domMeas)

        # BDF2
        exact.set_time(t_curr + dt)
        with TaskManager():
            f.Assemble()
            g.Assemble()

        rhs.data = F + (1.0-alpha)/(alpha*dt)*M*diff - A2bdf2 * U

        with TaskManager():
            solvers.MinRes(mat=Abdf2, pre=C, rhs=rhs, sol=diff, initialize=True, maxsteps=minres_iter, tol=1e-12,
                           printrates=printrates)
            U.data += diff
            renormalize(Q, mesh, ds, gfp, domMeas)

        t_curr += dt

        fold.data = f.vec

        if logs:
            print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        with TaskManager():
            l2p, h1p = errors_scal(mesh, ds, Pmat, gfp, coef_p)
            l2u, h1u = errors_vec(mesh, ds, Pmat, gfu, coef_u)
        mass_append(keys=keys, vals=[t_curr, l2u, h1u, l2p, h1p], **out_errs)

        if vtk_out:
            with TaskManager():
                vtk.Do(time=t_curr)
        i += 1

    if logs:
        print("")
        end = time.perf_counter()
        print(f" Time elapsed: {end - start: .5f} s")

    return V.ndof + Q.ndof, out_errs['ts'], out_errs['l2us'], out_errs['h1us'], out_errs['l2ps'], out_errs['h1ps']


def navier_stokes(mesh, exact, dt, tfinal, order, linear_solver_params, vtk_out=None, logs=True, printrates=False):
    """
    Solves Unsteady Navier-Stokes problem on a provided mesh using P_k-P_{k-1} Taylor-Hood pair.
    The initial data and RHS needs to be specified in a dictionary exact. VTK output can be provided if enabled.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma and is (ideally) refined around it.
        exact: exact.Exact
            Exact solution object, see exact.py and fixed_surface_poisson.py
        dt: float
            Time step size.
        tfinal: float
            Final time in the simulation.
        order: int
            Polynomial order for FEM for velocity.
        linear_solver_params: dict
            A dictionary with number of iterations for linear solvers.
        vtk_out: str
            String to be appended to the name of the VTK file.
        logs: bool
            Flag that indicates if logs are to be printed.
        printrates: bool
            Flag that indicates if linear solver residuals are to be printed.

    Returns:
        V.ndof + Q.ndof: int
            Number of degrees of freedom of the problem.
        ts: List[float]
            Discrete times t_n at which problem was solved.
        l2us: List[float]
            L^2-error of the velocity over Gamma at each time t_n
        h1us: List[float]
            H^1-error of the velocity over Gamma at each time t_n
        l2ps: List[float]
            L^2-error of the pressure over Gamma at each time t_n
        h1ps: List[float]
            H^1-error of the pressure over Gamma at each time t_n
    """
    if order < 3:
        precond_spd = "bddc"
        cg_iter = linear_solver_params['bddc_cg_iter']
    else:
        precond_spd = "local"
        cg_iter = linear_solver_params['jacobi_cg_iter']
    precond_nonsym = 'local'
    gmres_iter_inner = linear_solver_params['gmres_iter_inner']
    gmres_iter_outer = linear_solver_params['gmres_iter_outer']

    # unpack exact
    param_rho = exact.params['rho']
    param_mu = exact.params['mu']

    coef_phi = CoefficientFunction(exact.cfs["phi"]).Compile()
    coef_u = CoefficientFunction((exact.cfs["u1"], exact.cfs["u2"], exact.cfs["u3"])).Compile()
    coef_p = CoefficientFunction(exact.cfs['p']).Compile()
    coef_f = CoefficientFunction((exact.cfs["f1"], exact.cfs["f2"], exact.cfs["f3"])).Compile()
    coef_g = CoefficientFunction(exact.cfs['g']).Compile()

    # Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(coef_phi)
    lset_approx = lsetmeshadap.lset_p1

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
    n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=coef_phi, vel_space=VPhk)

    # declare the integration domains
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ci.GetElementsOfType(IF), deformation=deformation)
    dX = dx(definedonelements=ci.GetElementsOfType(IF), deformation=deformation)

    domMeas = get_dom_measure(Q, mesh, ds)

    # define projection matrix
    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)

    exact.set_time(-dt)
    mesh.SetDeformation(deformation)
    gfu_prev.Set(coef_u)
    mesh.UnsetDeformation()

    exact.set_time(0.0)
    mesh.SetDeformation(deformation)
    gfu.Set(coef_u)
    gfp.Set(coef_p)
    mesh.UnsetDeformation()

    gfu_approx.Set(2. * gfu - gfu_prev)
    bdf_coeff = 3./2

    # BI-LINEAR FORMS
    bilinear_form_args = {'param_rho': param_rho, 'param_mu': param_mu, 'bdf_coeff': bdf_coeff, 'dt': dt,
                          'gfu_approx': gfu_approx}

    m, a, ap, pd, b, c, sq, f, g = define_forms(
        eq_type='ns',
        V=V, Q=Q,
        n=n, Pmat=Pmat, Hmat=Hmat, n_k=n_k,
        coef_f=coef_f, coef_g=coef_g,
        ds=ds, dX=dX, **bilinear_form_args
    )

    start = time.perf_counter()
    with TaskManager():
        presq = Preconditioner(sq, precond_spd)
        prepd = Preconditioner(pd, precond_spd)
        prea = Preconditioner(a, precond_nonsym)

        assemble_forms([b, c, sq, m, pd])

    if logs:
        print(f"{bcolors.OKCYAN}System assembled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    # LINEAR SOLVER

    invsq = CGSolver(sq.mat, presq.mat, maxsteps=cg_iter, precision=1e-4)
    invpd = CGSolver(pd.mat, prepd.mat, maxsteps=cg_iter, precision=1e-4)

    U = BlockVector([gfu.vec, gfp.vec])

    # TIME MARCHING

    t_curr = 0.0  # time counter within one block-run

    if vtk_out:
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, deformation, gfu, gfp, coef_u, coef_p],
                        names=["lset_p1", "deform", "u", "p", "coef_u", "coef_p"],
                        filename=f"./output/vtk_out/fixed_surface_steady_stokes_p{order}_{exact.name}_{vtk_out}_mu={param_mu}",
                        subdivision=0)
        vtk.Do(time=0.0)

    rhs = BlockVector([f.vec.CreateVector(),
                       g.vec.CreateVector()])

    diffu = 0.0 * gfu.vec.CreateVector()
    diffp = 0.0 * gfp.vec.CreateVector()
    diff = BlockVector([diffu, diffp])

    keys = ['ts', 'l2us', 'h1us', 'l2ps', 'h1ps']
    out_errs = {'ts': [], 'l2us': [], 'h1us': [], 'l2ps': [], 'h1ps': []}

    with TaskManager():
        l2p, h1p = errors_scal(mesh, ds, Pmat, gfp, coef_p)
        l2u, h1u = errors_vec(mesh, ds, Pmat, gfu, coef_u)
    mass_append(keys=keys, vals=[t_curr, l2u, h1u, l2p, h1p], **out_errs)

    start = time.perf_counter()

    while t_curr < tfinal - 0.5 * dt:
        exact.set_time(t_curr + dt)
        gfu_approx.Set(2.0 * gfu - gfu_prev)
        with TaskManager():
            assemble_forms([f, g, a, ap])

            inva = GMRESSolver(a.mat, prea.mat, maxsteps=gmres_iter_inner, precision=1e-4)
            invms = invsq @ ap.mat @ invpd

            A = BlockMatrix([[a.mat, b.mat.T],
                             [b.mat, -c.mat]])

            C = BlockMatrix([[inva, inva @ b.mat.T @ invms],
                             [None, -invms]])

            F = BlockVector([f.vec + m.mat * (2.0/dt * gfu.vec - 0.5/dt * gfu_prev.vec),
                             g.vec])

            rhs.data = F - A * U

            gfu_prev.Set(gfu)

            solvers.GMRes(A=A, b=rhs, pre=C, x=diff, printrates=printrates, maxsteps=gmres_iter_outer, reltol=1e-12)
            U += diff

            renormalize(Q, mesh, ds, gfp, domMeas)

        t_curr += dt

        if logs:
            print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        with TaskManager():
            l2p, h1p = errors_scal(mesh, ds, Pmat, gfp, coef_p)
            l2u, h1u = errors_vec(mesh, ds, Pmat, gfu, coef_u)
        mass_append(keys=keys, vals=[t_curr, l2u, h1u, l2p, h1p], **out_errs)

        if vtk_out:
            with TaskManager():
                vtk.Do(time=t_curr)

    if logs:
        print("")
        end = time.perf_counter()
        print(f" Time elapsed: {end - start: .5f} s")

    return V.ndof + Q.ndof, out_errs['ts'], out_errs['l2us'], out_errs['h1us'], out_errs['l2ps'], out_errs['h1ps']


def moving_ns(mesh, exact, dt, tfinal, order, time_order, band,
              linear_solver_params, vtk_out=None, logs=True, printrates=False):
    """
    Solves evolving-surface Tangential Navier-Stokes on a provided mesh using GMRes. Intended to be used with zero RHS
    of the continuity equation (i.e. g=0). Otherwise might be unstable.
    The initial data and RHS needs to be specified in an object exact. VTK output can be provided if enabled.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma and is (ideally) refined around it.
        exact: exact.Exact
            Exact solution object, see exact.py and fixed_surface_poisson.py
        dt: float
            Time step size.
        tfinal: float
            Final time in the simulation.
        order: int
            Polynomial order for FEM for velocity.
        time_order: int
            Order of time discretization (BDF in this case).
        band: float
            Size of the band around levelset, the distance metric is in terms of the levelset function.
        linear_solver_params: dict
            A dictionary with number of iterations for linear solvers.
        vtk_out: str
            String to be appended to the name of the VTK file.
        logs: bool
            Flag that indicates if logs are to be printed.
        printrates: bool
            Flag that indicates if linear solver residuals are to be printed.
    Returns:
        np.mean(dofs): float
            Mean number of dofs per time step.
        ts: List[float]
            Discrete times t_n at which problem was solved.
        l2us: List[float]
            List of L^2-errors in velocity for each t_n.
        h1us: List[float]
            List of H^1-errors in velocity for each t_n.
        l2ps: List[float]
            List of L^2-errors in pressure for each t_n.
        h1ps: List[float]
            List of H^1-errors in pressure for each t_n.
    """
    if order < 3:
        precond_name = "bddc"
        cg_iter = linear_solver_params['bddc_cg_iter']
    else:
        precond_name = "local"
        cg_iter = linear_solver_params['jacobi_cg_iter']
    gmres_iter_outer = linear_solver_params['gmres_iter_outer']

    bdf_coeffs = [[1, -1],
                  [3 / 2, -2, 1 / 2],
                  [11 / 6, -3, 3 / 2, -1 / 3]]

    bdf_coeff = bdf_coeffs[time_order - 1]

    # unpack exact
    param_rho = exact.params['rho']
    param_mu = exact.params['mu']

    coef_phi = CoefficientFunction(exact.cfs["phi"]).Compile()
    coef_wN = CoefficientFunction(exact.cfs["wN"]).Compile()
    coef_u = CoefficientFunction((exact.cfs["u1"], exact.cfs["u2"], exact.cfs["u3"])).Compile()
    coef_p = CoefficientFunction(exact.cfs['p']).Compile()
    coef_f = CoefficientFunction((exact.cfs["f1"], exact.cfs["f2"], exact.cfs["f3"])).Compile()
    coef_g = CoefficientFunction(exact.cfs['g']).Compile()

    # Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(coef_phi)
    lset_approx = GridFunction(H1(mesh, order=1, dirichlet=[]))
    ba_IF = BitArray(mesh.ne)
    ba_IF_band = BitArray(mesh.ne)

    update_geometry(mesh, coef_phi, lset_approx, band, ba_IF, ba_IF_band)

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

    keys = ['ts', 'l2us', 'h1us', 'l2ps', 'h1ps']
    out_errs = {'ts': [], 'l2us': [], 'h1us': [], 'l2ps': [], 'h1ps': []}

    if vtk_out:
        gfu_out = GridFunction(V)
        gfp_out = GridFunction(Q)
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, coef_phi, gfu_out, gfp_out, coef_u, coef_p],
                        names=["P1-levelset", "phi", "u", "uSol", "p", "pSol"],
                        filename=f"./output/vtk_out/fixed_surface_steady_stokes_p{order}_{exact.name}_{vtk_out}_mu={param_mu}",
                        subdivision=0)

    start = time.perf_counter()

    set_ic(
        mesh=mesh, V=V, order=order, gfu_prevs=gfu_prevs, exact=exact, dt=dt,
        lsetmeshadap=lsetmeshadap, lset_approx=lset_approx, band=band, ba_IF=ba_IF, ba_IF_band=ba_IF_band,
        n=n, Pmat=Pmat, rho_u=rho_u, tau=tau, ds=ds, dX=dX, precond_name=precond_name, cg_iter=cg_iter
    )

    if logs:
        print(f"{bcolors.OKGREEN}IC for BDF{time_order} initialized ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    dofs = []
    # TIME MARCHING
    exact.set_time(0.0)
    t_curr = 0.0

    if vtk_out:
        gfu_out.Set(gfu_prevs[0])
        gfp_out.Set(coef_p)
        vtk.Do(time=t_curr)

    i = 1

    start = time.perf_counter()

    time_assembly = 0.0
    time_solver = 0.0

    while t_curr < tfinal + dt/2:
        exact.set_time(t_curr + dt)
        t_curr += dt
        with TaskManager():
            deformation = lsetmeshadap.CalcDeformation(coef_phi)

            update_geometry(mesh, coef_phi, lset_approx, band, ba_IF, ba_IF_band)

            VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
            QG = Compress(Q, GetDofsOfElements(Q, ba_IF))
            dofs.append(VG.ndof + QG.ndof)

            # helper grid functions
            n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=coef_phi, vel_space=V)

        gfu_approx = GridFunction(VG)
        if time_order == 1:
            gfu_approx.Set(Pmat * gfu_prevs[0])
        elif time_order == 2:
            gfu_approx.Set(2 * Pmat * gfu_prevs[0] - Pmat * gfu_prevs[1])
        else:
            gfu_approx.Set(3 * Pmat * gfu_prevs[0] - 3 * Pmat * gfu_prevs[1] + Pmat * gfu_prevs[2])

        a, ap, pd, sq, b, c, f, g = define_forms_moving_ns(
            VG=VG, QG=QG,
            bdf_coeff=bdf_coeff, dt=dt,
            n=n, Pmat=Pmat, n_k=n_k, Hmat=Hmat,
            tau=tau, rho_u=rho_u, rho_p=rho_p,
            ds=ds, dX=dX, dX2=dX2,
            param_rho=param_rho, param_mu=param_mu, coef_wN=coef_wN, coef_f=coef_f, coef_g=coef_g,
            gfu_approx=gfu_approx, gfu_prevs=gfu_prevs,
            solver='gmres'
        )

        with TaskManager():
            pre_a = Preconditioner(a, precond_name)
            pre_pd = Preconditioner(pd, precond_name)
            pre_sq = Preconditioner(sq, precond_name)

            start_assembly = time.perf_counter()
            assemble_forms([a, ap, pd, sq, b, c, f, g])
            time_assembly += (time.perf_counter() - start_assembly)

            K = BlockMatrix([[a.mat, b.mat.T],
                             [b.mat, -c.mat]])

            inva = CGSolver(a.mat, pre_a.mat, maxsteps=cg_iter, precision=1e-6)
            invpd = CGSolver(pd.mat, pre_pd.mat, maxsteps=cg_iter, precision=1e-6)
            invsq = CGSolver(sq.mat, pre_sq.mat, maxsteps=cg_iter, precision=1e-6)
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

            start_solver = time.perf_counter()
            solvers.GMRes(A=K, b=rhs, pre=C, x=sol, printrates=printrates, maxsteps=gmres_iter_outer, reltol=1e-12)
            time_solver += (time.perf_counter() - start_solver)

            # making numerical pressure mean zero
            renormalize(QG, mesh, ds, gfp)

            if vtk_out:
                gfu_out.Set(gfu)
                gfp_out.Set(gfp)
                vtk.Do(time=t_curr)

            l2p, h1p = errors_scal(mesh, ds, Pmat, gfp, coef_p)
            l2u, h1u = errors_vec(mesh, ds, Pmat, gfu, coef_u)

            gfu_prevs[0].Set(gfu)
            mass_append(keys=keys, vals=[t_curr, l2u, h1u, l2p, h1p], **out_errs)

        if logs:
            print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        i += 1

    if logs:
        print("")
        end = time.perf_counter()
        print(f" Time elapsed: {end - start: .5f} s")
        print(f"{bcolors.OKCYAN}Time in assembly:        {time_assembly:.5f} s.{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Time in solver:          {time_solver:.5f} s.{bcolors.ENDC}")

    return np.mean(dofs), out_errs['ts'], out_errs['l2us'], out_errs['h1us'], out_errs['l2ps'], out_errs['h1ps']


def moving_ns_direct(mesh, exact, dt, tfinal, order, time_order, band,
                     vtk_out=None, logs=True):
    """
    Solves evolving-surface Tangential Navier-Stokes on a provided mesh using GMRes. Intended to be used with zero RHS
    of the continuity equation (i.e. g=0). Otherwise might be unstable.
    The initial data and RHS needs to be specified in an object exact. VTK output can be provided if enabled.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma and is (ideally) refined around it.
        exact: exact.Exact
            Exact solution object, see exact.py and fixed_surface_poisson.py
        dt: float
            Time step size.
        tfinal: float
            Final time in the simulation.
        order: int
            Polynomial order for FEM for velocity.
        time_order: int
            Order of time discretization (BDF in this case).
        band: float
            Size of the band around levelset, the distance metric is in terms of the levelset function.
        vtk_out: str
            String to be appended to the name of the VTK file.
        logs: bool
            Flag that indicates if logs are to be printed.
    Returns:
        np.mean(dofs): float
            Mean number of dofs per time step.
        ts: List[float]
            Discrete times t_n at which problem was solved.
        l2us: List[float]
            List of L^2-errors in velocity for each t_n.
        h1us: List[float]
            List of H^1-errors in velocity for each t_n.
        l2ps: List[float]
            List of L^2-errors in pressure for each t_n.
        h1ps: List[float]
            List of H^1-errors in pressure for each t_n.
    """
    bdf_coeffs = [[1, -1],
                  [3 / 2, -2, 1 / 2],
                  [11 / 6, -3, 3 / 2, -1 / 3]]

    bdf_coeff = bdf_coeffs[time_order - 1]

    # unpack exact
    param_rho = exact.params['rho']
    param_mu = exact.params['mu']

    coef_phi = CoefficientFunction(exact.cfs["phi"]).Compile()
    coef_wN = CoefficientFunction(exact.cfs["wN"]).Compile()
    coef_u = CoefficientFunction((exact.cfs["u1"], exact.cfs["u2"], exact.cfs["u3"])).Compile()
    coef_p = CoefficientFunction(exact.cfs['p']).Compile()
    coef_f = CoefficientFunction((exact.cfs["f1"], exact.cfs["f2"], exact.cfs["f3"])).Compile()
    coef_g = CoefficientFunction(exact.cfs['g']).Compile()

    # MESH
    # Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(coef_phi)
    lset_approx = GridFunction(H1(mesh, order=1, dirichlet=[]))
    ba_IF = BitArray(mesh.ne)
    ba_IF_band = BitArray(mesh.ne)

    update_geometry(mesh, coef_phi, lset_approx, band, ba_IF, ba_IF_band)

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

    keys = ['ts', 'l2us', 'h1us', 'l2ps', 'h1ps']
    out_errs = {'ts': [], 'l2us': [], 'h1us': [], 'l2ps': [], 'h1ps': []}

    if vtk_out:
        gfu_out = GridFunction(V)
        gfp_out = GridFunction(Q)
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, coef_phi, gfu_out, gfp_out, coef_u, coef_p],
                        names=["P1-levelset", "phi", "u", "uSol", "p", "pSol"],
                        filename=f"./output/vtk_out/fixed_surface_steady_stokes_p{order}_{exact.name}_{vtk_out}_mu={param_mu}",
                        subdivision=0)

    start = time.perf_counter()

    set_ic(
        mesh=mesh, V=V, order=order, gfu_prevs=gfu_prevs, exact=exact, dt=dt,
        lsetmeshadap=lsetmeshadap, lset_approx=lset_approx, band=band, ba_IF=ba_IF, ba_IF_band=ba_IF_band,
        n=n, Pmat=Pmat, rho_u=rho_u, tau=tau, ds=ds, dX=dX
    )

    if logs:
        print(f"{bcolors.OKGREEN}IC for BDF{time_order} initialized ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    dofs = []
    # TIME MARCHING
    exact.set_time(0.0)
    t_curr = 0.0

    if vtk_out:
        gfu_out.Set(gfu_prevs[0])
        gfp_out.Set(exact.p)
        vtk.Do(time=t_curr)

    i = 1

    l2err_old = 0.0

    start = time.perf_counter()

    time_assembly = 0.0
    time_solver = 0.0

    while t_curr < tfinal + dt/2:
        exact.set_time(t_curr + dt)
        with TaskManager():
            deformation = lsetmeshadap.CalcDeformation(coef_phi)

            update_geometry(mesh, coef_phi, lset_approx, band, ba_IF, ba_IF_band)

            VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
            QG = Compress(Q, GetDofsOfElements(Q, ba_IF))
            XG = FESpace([VG, QG])
            dofs.append(XG.ndof)

            # helper grid functions
            n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=coef_phi, vel_space=V)

        gfu_approx = GridFunction(VG)
        if time_order == 1:
            gfu_approx.Set(Pmat * gfu_prevs[0])
        elif time_order == 2:
            gfu_approx.Set(2 * Pmat * gfu_prevs[0] - Pmat * gfu_prevs[1])
        else:
            gfu_approx.Set(3 * Pmat * gfu_prevs[0] - 3 * Pmat * gfu_prevs[1] + Pmat * gfu_prevs[2])

        a, f, = define_forms_moving_ns(
            VG=VG, QG=QG,
            bdf_coeff=bdf_coeff, dt=dt,
            n=n, Pmat=Pmat, n_k=n_k, Hmat=Hmat,
            tau=tau, rho_u=rho_u, rho_p=rho_p,
            ds=ds, dX=dX, dX2=dX2,
            param_rho=param_rho, param_mu=param_mu, coef_wN=coef_wN, coef_f=coef_f, coef_g=coef_g,
            gfu_approx=gfu_approx, gfu_prevs=gfu_prevs,
            solver='direct', XG=XG
        )

        with TaskManager():
            start_assembly = time.perf_counter()
            assemble_forms([a, f])
            time_assembly += (time.perf_counter() - start_assembly)

            gf = GridFunction(XG)

        start_solver = time.perf_counter()
        gf.vec.data = a.mat.Inverse(freedofs=XG.FreeDofs(), inverse="umfpack") * f.vec
        time_solver += (time.perf_counter() - start_solver)

        gfu = gf.components[0]

        l2err = sqrt(Integrate(InnerProduct(Pmat * (gfu - gfu_prevs[0]), Pmat * (gfu - gfu_prevs[0])) * ds, mesh))

        if i > 1 and l2err > 2 * l2err_old:
            continue

        for j in range(time_order-1):
            gfu_prevs[-1 - j].vec.data = gfu_prevs[-2 - j].vec

        gfp = gf.components[1]
        # making numerical pressure mean zero
        renormalize(QG, mesh, ds, gfp)

        if vtk_out:
            gfu_out.Set(gfu)
            gfp_out.Set(gfp)
            vtk.Do(time=t_curr+dt)

        l2p, h1p = errors_scal(mesh, ds, Pmat, gfp, exact.p)
        l2u, h1u = errors_vec(mesh, ds, Pmat, gfu, exact.u)

        gfu_prevs[0].Set(gfu)
        mass_append(keys=keys, vals=[t_curr, l2u, h1u, l2p, h1p], **out_errs)

        if logs:
            print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        l2err_old = l2err
        t_curr += dt
        i += 1

    if logs:
        print("")
        end = time.perf_counter()
        print(f" Time elapsed: {end - start: .5f} s")
        print(f"{bcolors.OKCYAN}Time in assembly:        {time_assembly:.5f} s.{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Time in solver:          {time_solver:.5f} s.{bcolors.ENDC}")

    return np.mean(dofs), out_errs['ts'], out_errs['l2us'], out_errs['h1us'], out_errs['l2ps'], out_errs['h1ps']