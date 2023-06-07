# ------------------------------ LOAD LIBRARIES -------------------------------
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from xfem.lsetcurv import *
from ngsolve import solvers
from ngsolve import TaskManager
import time
import sys
import numpy as np
from utils import bcolors, assemble_forms, mass_append, errors_scal, update_geometry


# HELPERS

def define_forms(eq_type, V, n, Pmat, coef_f, ds, dX, **args):
    """
    Routine that defines bilinear and linear forms for the poisson and the diffusion equation.
    Args:
        eq_type: str
            Type of PDE: "poisson" or "diffusion" (default)
        V: Compressed H1 space
            TraceFEM space for u
        n: Vector-valued GridFunction
            Discrete normal vector
        Pmat: Tensor-valued GridFunction
            Discrete projection matrix
        rhsf: CoefficientFunction
            Right-hand side of the PDE
        ds: xfem.dCul
            Element of area on Gamma
        dX: ngsolve.utils.dx
            Element of bulk volume
        **args:
            Parameters that are problem-dependent, like
            args['alpha']: float
                Coefficient in front of the mass term
            args['nu']: float
                Coefficient of kinematic viscocity
            args['dt']: float
                Time step size
            args['gfu_approx']: Vector-valued GridFunction
                Extrapolation of u at t=t_{n+1} for linearization of the convection term in Navier-Stokes

    Returns:
        A tuple of bilinear and linear problems relevant to the problem
    """
    u, v = V.TnT()
    h = specialcf.mesh_size

    if eq_type == 'poisson':
        # penalization parameters
        rho_u = 1.0 / h
        param_alpha = args['param_alpha']
        param_nu = args['param_nu']

        # a_h part
        a = BilinearForm(V, symmetric=True)
        a += param_alpha * u * v * ds
        a += param_nu * InnerProduct(Pmat * grad(u), Pmat * grad(v)) * ds
        # normal gradient volume stabilization of the velocity
        a += param_nu * rho_u * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX

        f = LinearForm(V)
        f += coef_f * v * ds

        return a, f
    else:
        # penalization parameters
        rho_u = 1.0 / h
        param_alpha = args['param_alpha']
        param_nu = args['param_nu']
        tr_bdf2_param = args['tr_bdf2_param']
        dt = args['dt']
        stab_type = args['stab_type']

        m = BilinearForm(V, symmetric=True)  # mass
        d = BilinearForm(V, symmetric=True)  # total_stab_tests_diffusion
        a = BilinearForm(V, symmetric=True)  # mass-total_stab_tests_diffusion

        # mass part
        m += param_alpha * u * v * ds
        a += param_alpha * 2.0 / (tr_bdf2_param * dt) * u * v * ds

        if stab_type in ['new', 'total']:
            # stabilizing mass part
            m += param_alpha * h * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX
            a += param_alpha * 2.0 / (tr_bdf2_param * dt) * h * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX

        if stab_type in ['old', 'total']:
            # stabilizing diffusion part
            d += param_nu * rho_u * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX
            a += param_nu * rho_u * InnerProduct(grad(u), n) * InnerProduct(grad(v), n) * dX

        # diffusion part
        a += param_nu * InnerProduct(Pmat * grad(u), Pmat * grad(v)) * ds
        d += param_nu * InnerProduct(Pmat * grad(u), Pmat * grad(v)) * ds

        f = LinearForm(V)
        f += coef_f * v * ds

        return m, d, a, f


def set_ic(mesh, V, gfu_prevs, exact, dt, lsetmeshadap, lset_approx, band, ba_IF, ba_IF_band, n, Pmat, rho_u, ds, dX):
    """
    A routine that sets initial condition for BDFk method with normal extension on a narrowband around the surface.
    Might be unified with analogous routine for evolving-surface NS in the future.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma and is (ideally) refined around it.
        V: H1
            Ambient FE space for velocity.
        exact: Exact
            Exact solution object, see stokes_solvers.py file.
        gfu_prevs: List[GridFunction(V)]
            List of GridFunction's defined on the ambient velocity space. Should contain k grid function for the BDFk
            method.
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
            Normal gradient stabilization parameter for u.
        ds: xfem.dCul
            Element of area on Gamma.
        dX: ngsolve.utils.dx
            Element of bulk volume in the band around surface.

    Returns:

    """
    time_order = len(gfu_prevs)
    coef_phi = exact.cfs['phi']
    coef_fel = exact.cfs['fel']
    for j in range(time_order):
        # fix levelset
        exact.set_time(-j * dt)
        deformation = lsetmeshadap.CalcDeformation(coef_phi)

        update_geometry(mesh, coef_phi, lset_approx, band, ba_IF, ba_IF_band)

        # solve elliptic problem on a fixed surface to get u with normal extension

        VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
        gfu_el = GridFunction(VG)
        u_el, v_el = VG.TnT()

        a_el = BilinearForm(VG, symmetric=True)
        a_el += (u_el * v_el + InnerProduct(Pmat * grad(u_el), Pmat * grad(v_el))) * ds
        a_el += rho_u * (n * grad(u_el)) * (n * grad(v_el)) * dX

        f_el = LinearForm(VG)
        f_el += coef_fel * v_el * ds

        with TaskManager():
            c_el = Preconditioner(a_el, "bddc")
            a_el.Assemble()
            f_el.Assemble()

            solvers.CG(mat=a_el.mat, rhs=f_el.vec, pre=c_el.mat, sol=gfu_el.vec, printrates=False)
            sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

            gfu_prevs[j].Set(gfu_el)


# SOLVERS
def poisson(mesh, exact, order, linear_solver_params, vtk_out=None, logs=True, printrates=False):
    """
    Solves Poisson equation (with an added mass term to allow non-mean-zero right-hand-sides) on a provided mesh.
    The initial data and the RHS needs to be specified in a dictionary exact. VTK output can be provided if enabled.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma and is (ideally) refined around it.
        exact: exact.Exact
            Exact solution object, see exact.py and fixed_surface_poisson.py
        linear_solver_params: dict
            A dictionary with number of iterations for linear solvers.
        order: int
            Polynomial order for FEM, default 1.
        vtk_out: str
            String to be appended to the name of the VTK file.
        logs: bool
            Flag that indicates if logs are to be printed.
        printrates: bool
            Flag that indicates if linear solver residuals are to be printed.

    Returns:
        V.ndof: int
            Number of degrees of freedom of the problem.
        l2u: float
            L^2-error of the solution over Gamma
        h1u: float
            H^1-error of the solution over Gamma
    """
    if order < 3:
        precond_name = "bddc"
        cg_iter = linear_solver_params['bddc_cg_iter']
    else:
        precond_name = "local"
        cg_iter = linear_solver_params['jacobi_cg_iter']

    # unpack exact
    param_nu = exact.params['nu']
    param_alpha = exact.params['alpha']

    coef_phi = CoefficientFunction(exact.cfs["phi"]).Compile()
    coef_u = CoefficientFunction(exact.cfs["u"]).Compile()
    coef_f = CoefficientFunction(exact.cfs["f"]).Compile()

    # LEVELSET ADAPTATION
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(coef_phi)
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

    # define projection matrix
    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)

    # bilinear forms:
    bilinear_form_args = {'param_alpha': param_alpha, 'param_nu': param_nu}
    a, f = define_forms(eq_type='poisson', V=V, n=n, Pmat=Pmat, coef_f=coef_f, ds=ds, dX=dX, **bilinear_form_args)

    start = time.perf_counter()
    with TaskManager():
        prea = Preconditioner(a, precond_name)
        assemble_forms([a, f])

    if logs:
        print(f"{bcolors.OKCYAN}System assembled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    # LINEAR SOLVER

    start = time.perf_counter()
    with TaskManager():
        solvers.CG(mat=a.mat, pre=prea.mat, rhs=f.vec, sol=gfu.vec, maxsteps=cg_iter, initialize=True, tol=1e-12,
                   printrates=printrates)
    sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning
    if logs:
        print(f"{bcolors.OKBLUE}System solved    ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    # ERRORS

    with TaskManager():
        l2u, h1u = errors_scal(mesh, ds, Pmat, gfu, coef_u)

    if vtk_out:
        with TaskManager():
            vtk = VTKOutput(ma=mesh,
                            coefs=[lset_approx, deformation, gfu, coef_u],
                            names=["lset_p1", "deform", "u", "coef_u"],
                            filename=f"./output/vtk_out/fixed_surface_poisson_p{order}_{exact.name}_{vtk_out}", subdivision=0)
            vtk.Do()

    return V.ndof, l2u, h1u


def diffusion(mesh, exact, dt, tfinal, order, linear_solver_params, vtk_out=None,
              logs=True, printrates=False, stab_type='old'):
    """
    Solves diffusion equation on a provided mesh. The initial data and RHS needs to be specified in a dictionary exact.
    VTK output can be provided if enabled.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma and is (ideally) refined around it.
        exact: exact.Exact
            Exact solution object, see exact.py and fixed_surface_diffusion_test.py
            If exact.cfs['fel'] is defined, use difference initialization of IC via auxiliary problem.
        dt: float
            Time step size.
        tfinal: float
            Final time in the simulation.
        order: int
            Polynomial order for FEM.
        linear_solver_params: dict
            A dictionary with number of iterations for linear solvers.
        vtk_out: str
            String to be appended to the name of the VTK file.
        logs: bool
            Flag that indicates if logs are to be printed.
        printrates: bool
            Flag that indicates if linear solver residuals are to be printed.
        stab_type: str
            Type of stabilization. The standard one is what we call here 'old', i.e. normal gradient in the bulk.
            We also offer two other (more experimental) types of stabilizations:
                - 'new': where we stabilize the mass term instead of the diffusion term with a scaling factor h.
                - 'total': where we apply both 'old' and 'new' stabilizations.
            The motivation behind the other two stabilizations is that they have better conditioning properties
            when dt -> 0, h fixed. But again, consistency analysis of these two schemes has not been conducted yet
            (as of May 2023).

    Returns:
        V.ndof: int
            Number of degrees of freedom of the problem.
        out_errs['ts']: List[float]
            Discrete times t_n at which problem was solved.
        out_errs['l2us']: List[float]
            List of L^2-error for each t_n.
        out_errs['h1us']: List[float]
            List of H^1-error for each t_n.
    """
    if order < 3:
        precond_name = "bddc"
        cg_iter = linear_solver_params['bddc_cg_iter']
    else:
        precond_name = "local"
        cg_iter = linear_solver_params['jacobi_cg_iter']

    # unpack exact
    param_alpha = exact.params['alpha']
    param_nu = exact.params['nu']

    coef_phi = CoefficientFunction(exact.cfs["phi"]).Compile()
    coef_u = CoefficientFunction(exact.cfs["u"]).Compile()
    coef_f = CoefficientFunction(exact.cfs["f"]).Compile()
    coef_fel = None
    if 'fel' in exact.cfs:
        coef_fel = CoefficientFunction(exact.cfs["fel"]).Compile()

    # LEVELSET ADAPTATION
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(coef_phi)
    lset_approx = lsetmeshadap.lset_p1

    ci = CutInfo(mesh, lset_approx)

    # FESpace
    Vh = H1(mesh, order=order, dirichlet=[])
    V = Compress(Vh, GetDofsOfElements(Vh, ci.GetElementsOfType(IF)))

    tr_bdf2_param = 2.0 - np.sqrt(2.0)

    # declare grid functions to store the solution
    gfu = GridFunction(V)

    # declare the integration domains
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ci.GetElementsOfType(IF), deformation=deformation)
    dX = dx(definedonelements=ci.GetElementsOfType(IF), deformation=deformation)

    # define projection matrix
    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)

    ### bilinear forms:
    bilinear_form_args = {'param_alpha': param_alpha, 'param_nu': param_nu,
                          'tr_bdf2_param': tr_bdf2_param, 'dt': dt, 'stab_type': stab_type}
    m, d, a, f = define_forms(eq_type='total_stab_tests_diffusion', V=V, n=n, Pmat=Pmat, coef_f=coef_f, ds=ds, dX=dX, **bilinear_form_args)

    start = time.perf_counter()
    with TaskManager():
        prea = Preconditioner(a, precond_name)
        assemble_forms([m, d, a, f])

    if logs:
        print(f"{bcolors.OKCYAN}System assembled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    ### LINEAR SOLVER

    diff = gfu.vec.CreateVector()
    fold = f.vec.CreateVector()
    rhs = f.vec.CreateVector()

    # TIME MARCHING

    t_curr = 0.0  # time counter within one block-run

    # IC
    exact.set_time(0.0)

    if coef_fel:
        # bad RHS
        gfu_el = GridFunction(V)
        bilinear_form_args = {'param_alpha': 1.0, "param_nu": 1.0}
        a_el, f_el = define_forms(eq_type='poisson', V=V, n=n, Pmat=Pmat, coef_f=coef_fel, ds=ds, dX=dX,
                                  **bilinear_form_args)
        start = time.perf_counter()
        with TaskManager():
            prea_el = Preconditioner(a_el, precond_name)
            assemble_forms([a_el, f_el])
            solvers.CG(mat=a_el.mat, pre=prea_el.mat, rhs=f_el.vec, sol=gfu_el.vec, maxsteps=cg_iter, initialize=True,
                       tol=1e-12, printrates=printrates)
        sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning
        gfu.Set(gfu_el)
        if logs:
            print(f"{bcolors.OKGREEN}IC computed      ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")
    else:
        mesh.SetDeformation(deformation)
        gfu.Set(coef_u)
        mesh.UnsetDeformation()

    if vtk_out:
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, deformation, gfu, coef_u],
                        names=["P1-levelset", "deform", "u", "uSol"],
                        filename=f"./output/vtk_out/fixed_surface_diffusion_p{order}_{exact.name}_{vtk_out}",
                        subdivision=0)
        vtk.Do(time=0.0)

    out_errs = {'ts': [], 'l2us': [], 'h1us': []}
    keys = ['ts', 'l2us', 'h1us']

    with TaskManager():
        l2u, h1u = errors_scal(mesh, ds, Pmat, gfu, coef_u)
    mass_append(keys=keys, vals=[t_curr, l2u, h1u], **out_errs)

    start = time.perf_counter()

    fold.data = f.vec
    i = 0
    while t_curr < tfinal - 0.5 * dt:
        exact.set_time(t_curr + tr_bdf2_param*dt)
        with TaskManager():
            f.Assemble()

        # TR
        rhs.data = fold + f.vec - 2 * d.mat * gfu.vec

        with TaskManager():
            solvers.CG(mat=a.mat, pre=prea.mat, rhs=rhs, sol=diff, initialize=True, maxsteps=cg_iter, tol=1e-12,
                       printrates=printrates)
            gfu.vec.data += diff
        sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

        # BDF2
        exact.set_time(t_curr + dt)
        with TaskManager():
            f.Assemble()

        rhs.data = f.vec + (1.0-tr_bdf2_param)/(tr_bdf2_param*dt) * m.mat * diff - d.mat * gfu.vec

        with TaskManager():
            solvers.CG(mat=a.mat, pre=prea.mat, rhs=rhs, sol=diff, initialize=True, maxsteps=cg_iter, tol=1e-12,
                       printrates=printrates)
            gfu.vec.data += diff
        sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

        t_curr += dt

        fold.data = f.vec

        if logs:
            print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        with TaskManager():
            l2u, h1u = errors_scal(mesh, ds, Pmat, gfu, coef_u)
        mass_append(keys=keys, vals=[t_curr, l2u, h1u], **out_errs)

        if vtk_out:
            with TaskManager():
                vtk.Do(time=t_curr)
        i += 1

    if logs:
        print("")
        end = time.perf_counter()
        print(f" Time elapsed: {end - start: .5f} s")

    return V.ndof, out_errs['ts'], out_errs['l2us'], out_errs['h1us']


def moving_diffusion(mesh, exact, dt, tfinal, order, time_order, band, linear_solver_params, vtk_out=None,
                     logs=True, printrates=False, stab_type='old'):
    """
    Solves evolving-surface diffusion equation on a provided mesh. The initial data and RHS needs to be specified in a
    object exact. VTK output can be provided if enabled.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma and is (ideally) refined around it.
        exact: exact.Exact
            Exact solution object, see exact.py and evolving_surface_diffusion.py
        dt: float
            Time step size.
        tfinal: float
            Final time in the simulation.
        order: int
            Polynomial order for FEM, default 1.
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
        stab_type: str
            Type of stabilization. The standard one is what we call here 'old', i.e. normal gradient in the bulk.
            We also offer two other (more experimental) types of stabilizations:
                - 'new': where we stabilize the mass term instead of the diffusion term with a scaling factor h.
                - 'total': where we apply both 'old' and 'new' stabilizations.
            The motivation behind the other two stabilizations is that they have better conditioning properties
            when dt -> 0, h fixed. But again, consistency analysis of these two schemes has not been conducted yet
            (as of May 2023).

    Returns:
        np.mean(dofs): float
            Mean number of dofs per time step
        ts: List[float]
            Discrete times t_n at which problem was solved.
        l2us: List[float]
            List of L^2-error for each t_n.
        h1us: List[float]
            List of H^1-error for each t_n.
    """
    if order < 3:
        precond_name = "bddc"
        gmres_iter = linear_solver_params['bddc_gmres_iter']
    else:
        precond_name = "local"
        gmres_iter = linear_solver_params['jacobi_gmres_iter']

    bdf_coeffs = [[1, -1],
                  [3 / 2, -2, 1 / 2],
                  [11 / 6, -3, 3 / 2, -1 / 3]]

    bdf_coeff = bdf_coeffs[time_order-1]

    # unpack exact
    param_alpha = exact.params['alpha']
    param_nu = exact.params['nu']

    coef_phi = exact.cfs['phi']
    coef_wN = CoefficientFunction((exact.cfs['w1'], exact.cfs['w2'], exact.cfs['w3']))
    coef_u = exact.cfs['u']
    coef_f = exact.cfs['f']
    coef_divGw = exact.cfs['divGw']
    coef_divGwT = exact.cfs['divGwT']

    exact.set_time(0.0)

    V = H1(mesh, order=order, dirichlet=[])

    # LEVELSET ADAPTATION
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(coef_phi)

    lset_approx = GridFunction(H1(mesh, order=1, dirichlet=[]))
    ba_IF = BitArray(mesh.ne)
    ba_IF_band = BitArray(mesh.ne)

    update_geometry(mesh, coef_phi, lset_approx, band, ba_IF, ba_IF_band)

    # define projection matrix
    ds = dCut(levelset=lset_approx, domain_type=IF, definedonelements=ba_IF, deformation=deformation)
    dX = dx(definedonelements=ba_IF_band, deformation=deformation)

    n = Normalize(grad(lset_approx))
    Pmat = Id(3) - OuterProduct(n, n)
    h = specialcf.mesh_size
    rho_u = 1./h
    rho_u_new = h

    gfu_prevs = [GridFunction(V, name=f"gru-{i}") for i in range(time_order)]

    if vtk_out:
        gfu_out = GridFunction(V)
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, gfu_out, coef_u],
                        names=["lset_p1", "u", "coef_u"],
                        filename=f"./output/vtk_out/evolving_surface_diffusion_p{order}_{exact.name}_{vtk_out}",
                        subdivision=0)

    keys = ['ts', 'l2us', 'h1us']
    out_errs = {'ts': [], 'l2us': [], 'h1us': []}

    start = time.perf_counter()

    # IC
    set_ic(
        mesh=mesh, V=V, gfu_prevs=gfu_prevs, exact=exact, dt=dt, lsetmeshadap=lsetmeshadap, lset_approx=lset_approx,
        band=band, ba_IF=ba_IF, ba_IF_band=ba_IF_band, n=n, Pmat=Pmat, rho_u=rho_u, ds=ds, dX=dX
    )

    if logs:
        print(f"{bcolors.OKGREEN}IC for BDF{time_order} initialized ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    # TIME MARCHING
    dofs = []

    exact.set_time(0.0)
    t_curr = 0.0

    if vtk_out:
        gfu_out.Set(gfu_prevs[0])
        vtk.Do(time=t_curr)

    i = 1

    start = time.perf_counter()

    time_assembly = 0.0
    time_solver = 0.0

    while t_curr < tfinal + dt/2:
        exact.set_time(t_curr + dt)
        with TaskManager():
            deformation = lsetmeshadap.CalcDeformation(coef_phi)
            update_geometry(mesh, coef_phi, lset_approx, band, ba_IF, ba_IF_band)

            VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
            dofs.append(VG.ndof)
            u, v = VG.TnT()

        a = BilinearForm(VG)
        a += param_alpha * bdf_coeff[0] * u * v * ds
        a += dt * (param_alpha / 2 * InnerProduct(Pmat * coef_wN, Pmat * grad(u)) * v -
                   param_alpha / 2 * InnerProduct(Pmat * coef_wN, Pmat * grad(v)) * u +
                   (coef_divGw - param_alpha / 2 * coef_divGwT) * u * v) * ds
        a += dt * param_nu * InnerProduct(Pmat * grad(u), Pmat * grad(v)) * ds
        if stab_type in ['old', 'total']:
            a += dt * rho_u * (n * grad(u)) * (n * grad(v)) * dX
        if stab_type in ['new', 'total']:
            a += param_alpha * bdf_coeff[0] * rho_u_new * (n * grad(u)) * (n * grad(v)) * dX

        f = LinearForm(VG)
        f += (dt * coef_f - sum([bdf_coeff[j+1] * gfu_prevs[j] for j in range(time_order)])) * v * ds

        with TaskManager():
            c = Preconditioner(a, precond_name)
            start_assembly = time.perf_counter()
            a.Assemble()
            f.Assemble()
            time_assembly += (time.perf_counter() - start_assembly)

            gfu = GridFunction(VG)

            start_solver = time.perf_counter()
            solvers.GMRes(A=a.mat, b=f.vec, pre=c.mat, x=gfu.vec, tol=1e-15, maxsteps=gmres_iter, printrates=printrates)
            time_solver += (time.perf_counter() - start_solver)

            for j in range(time_order-1):
                gfu_prevs[-1-j].vec.data = gfu_prevs[-2-j].vec
            gfu_prevs[0].Set(gfu)

            if vtk_out:
                gfu_out.Set(gfu)
                vtk.Do(time=t_curr)

        if logs:
            print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        t_curr += dt

        with TaskManager():
            l2u, h1u = errors_scal(mesh, ds, Pmat, gfu, coef_u)
        mass_append(keys=keys, vals=[t_curr, l2u, h1u], **out_errs)

        i += 1

    if logs:
        print("")
        end = time.perf_counter()
        print(f" Time elapsed: {end - start: .5f} s")
        print(f"{bcolors.OKCYAN}Time in assembly:        {time_assembly:.5f} s.{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Time in solver:          {time_solver:.5f} s.{bcolors.ENDC}")

    return np.mean(dofs), out_errs['ts'], out_errs['l2us'], out_errs['h1us']
