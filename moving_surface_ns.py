# ------------------------------ LOAD LIBRARIES -------------------------------
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from xfem.lsetcurv import *
from ngsolve import solvers
from ngsolve import TaskManager
import time
import sys
import numpy as np
from utils import renormalize, errors_scal, errors_vec, mass_append, \
    update_ba_IF_band, helper_grid_functions, assemble_forms


# EXACT SOLUTION CLASS

class Exact:
    """
    A class for exact solution of evolving-surface tangential Navier-Stokes. Its usage is motivated by the fact that
    it can store its own state (time). Might become an interface in later versions.
    """
    def __init__(self, mu, R, maxvel):
        """
        Default constructor that initializes time of the object.
        Args:
            mu: float
                Kinematic viscocity parameter.
            R: float
                Radius of the domain in terms of the distance function phi.
            maxvel: float
                Maximum norm of the velocity over space and simulation time.
        """
        self.t = Parameter(0.0)
        self.mu = Parameter(mu)
        self.R = Parameter(R)
        self.maxvel = maxvel

        self.phi = None
        self.wN = None
        self.u = None
        self.f = None
        self.g = None
        self.fel = None

    def set_params(self, phi, wN, u, p, f, fel, g):
        """
        Sets parameters of the exact solution
        Args:
            phi: CoefficientFunction
                Levelset function.
            wN: CoefficientFunction
                Normal component of the ambient velocity.
            u: Vector-valued CoefficientFunction
                Velocity.
            p: CoefficientFunction
                Pressure.
            f: Vector-valued CoefficientFunction
                RHS of the momentum equation.
            fel: Vector-valued CoefficientFunction
                RHS of the auxiliary problem for normal extension of the initial condition(s).
            g: CoefficientFunction
                RHS of the continuity equation.
        Returns:

        """
        self.phi = phi
        self.wN = wN
        self.u = u
        self.p = p
        self.f = f
        self.fel = fel
        self.g = g

    def set_time(self, tval):
        """
        Changes the time of the Exact solution object to tval.
        Args:
            tval: float
                New time of the exact solution object.
        Returns:

        """
        self.t.Set(tval)


# HELPERS


def update_geometry(mesh, phi, lset_approx, band, ba_IF, ba_IF_band):
    """
    Helper function which is a wrapper around ba_IF_band.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma and is (ideally) refined around it.
        phi: CoefficientFunction
            Levelset function
        lset_approx: P1 GridFunction on mesh.
            Variable for P1 approximation of the levelset.
        band: float
            Size of the band around levelset, the distance metric is in terms of the levelset function.
        ba_IF: BitArray
            Stores element numbers that are intersected by the surface.
        ba_IF_band: BitArray
            Stores element numbers that are in the narrowband around the surface.

    Returns:

    """
    InterpolateToP1(phi, lset_approx)
    ci = CutInfo(mesh, lset_approx)

    ba_IF.Clear()
    ba_IF |= ci.GetElementsOfType(IF)
    update_ba_IF_band(lset_approx, mesh, band, ba_IF_band)


def set_ic(mesh, V, order, gfu_prevs, exact, dt,
           lsetmeshadap, lset_approx, band, ba_IF, ba_IF_band,
           n, Pmat, rho_u, tau, ds, dX):
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
            Exact solution object, see moving_surface_ns.py file.
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

    Returns:

    """
    time_order = len(gfu_prevs)
    for j in range(time_order):
        # fix levelset
        exact.set_time(-j * dt)

        deformation = lsetmeshadap.CalcDeformation(exact.phi)

        # solve elliptic problem on a fixed surface to get u with normal extension
        update_geometry(mesh, exact.phi, lset_approx, band, ba_IF, ba_IF_band)
        VG = Compress(V, GetDofsOfElements(V, ba_IF_band))

        # helper grid functions
        n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=exact.phi, vel_space=V)

        gfu_el = GridFunction(VG)

        u_el, v_el = VG.TnT()

        a_el = BilinearForm(VG, symmetric=True)
        a_el += InnerProduct(Pmat * u_el, Pmat * v_el) * ds
        a_el += 2 * exact.mu * (InnerProduct(Pmat * Sym(grad(u_el)) * Pmat - (u_el * n) * Hmat,
                                             Pmat * Sym(grad(v_el)) * Pmat - (v_el * n) * Hmat)) * ds
        a_el += (tau * ((u_el * n_k) * (v_el * n_k))) * ds
        a_el += (rho_u * InnerProduct(grad(u_el) * n, grad(v_el) * n)) * dX

        f_el = LinearForm(VG)
        f_el += InnerProduct(exact.fel, Pmat * v_el) * ds

        with TaskManager():
            pre_a_el = Preconditioner(a_el, "bddc")

            a_el.Assemble()
            f_el.Assemble()

            solvers.CG(mat=a_el.mat, rhs=f_el.vec, pre=pre_a_el.mat, sol=gfu_el.vec, printrates=False)
            sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

            gfu_prevs[j].Set(gfu_el)


def define_forms(VG, QG,
                 bdf_coeff, dt,
                 n, Pmat, n_k, Hmat,
                 tau, rho_u, rho_p,
                 ds, dX, dX2,
                 exact, gfu_approx, gfu_prevs,
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
        exact: Exact
            Exact solution object, see moving_surface_ns.py file.
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
        a += bdf_coeff[0] / dt * InnerProduct(u, Pmat * v) * ds
        a += exact.wN * InnerProduct(Hmat * u, Pmat * v) * ds
        a += 0.5 * InnerProduct((Pmat * grad(u) * Pmat - (u * n) * Hmat) * gfu_approx, v) * ds
        a += (-0.5) * InnerProduct((Pmat * grad(v) * Pmat - (v * n) * Hmat) * gfu_approx, u) * ds
        a += (-0.5) * InnerProduct(exact.g * u, Pmat * v) * ds
        a += 2.0 * exact.mu * (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                                            Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        a += tau * InnerProduct(n_k, u) * InnerProduct(n_k, v) * ds
        a += rho_u * InnerProduct(grad(u) * n, grad(v) * n) * dX

        # pressure mass-convection-diffusion matrix
        ap = BilinearForm(QG, symmetric=False)
        # mass part
        ap += bdf_coeff[0] / dt * p * q * ds
        # total_stab_tests_diffusion
        ap += 2 * exact.mu * InnerProduct(Pmat * grad(p), Pmat * grad(q)) * ds
        # convection
        ap += InnerProduct(Pmat * grad(p), Pmat * gfu_approx) * q * ds
        # normal gradient in the bulk stabilization
        # SHOULD IT BE rho_p OR rho_u?
        ap += rho_p * ((grad(p) * n) * (grad(q) * n)) * dX2

        # pressure diffusion matrix
        pd = BilinearForm(QG, symmetric=True)
        # diffusion
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
        f += InnerProduct(exact.f, Pmat * v) * ds
        for j in range(time_order):
            f += (-1.0) * bdf_coeff[j + 1] / dt * InnerProduct(gfu_prevs[j], Pmat * v) * ds

        g = LinearForm(QG)
        g += (-1.0) * exact.g * q * ds

        return a, ap, pd, sq, b, c, f, g
    else:
        u, p = XG.TrialFunction()
        v, q = XG.TestFunction()

        a = BilinearForm(XG, symmetric=False)
        a += bdf_coeff[0] / dt * InnerProduct(u, Pmat * v) * ds
        a += exact.wN * InnerProduct(Hmat * u, Pmat * v) * ds
        a += 0.5 * InnerProduct((Pmat * grad(u) * Pmat - (u * n) * Hmat) * gfu_approx, v) * ds
        a += (-0.5) * InnerProduct((Pmat * grad(v) * Pmat - (v * n) * Hmat) * gfu_approx, u) * ds
        a += (-0.5) * InnerProduct(exact.g * u, Pmat * v) * ds
        a += 2.0 * exact.mu * (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                                      Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        a += tau * InnerProduct(n_k, u) * InnerProduct(n_k, v) * ds
        a += rho_u * InnerProduct(grad(u) * n, grad(v) * n) * dX

        a += InnerProduct(u, Pmat * grad(q)) * ds
        a += InnerProduct(v, Pmat * grad(p)) * ds

        a += (-1.0) * rho_p * (grad(p) * n) * (grad(q) * n) * dX2

        f = LinearForm(XG)
        f += InnerProduct(exact.f, Pmat * v) * ds
        for j in range(time_order):
            f += (-1.0) * bdf_coeff[j + 1] / dt * InnerProduct(gfu_prevs[j], Pmat * v) * ds
        f += (-1.0) * exact.g * q * ds

        return a, f


# SOLVERS

def moving_ns_direct(mesh, dt, order, tfinal, exact, band, time_order=1, out=False, fname=None):
    """
    Solves evolving-surface Tangential Navier-Stokes on a provided mesh using DIRECT SOLVER. This is a temporary
    workaround for the case of non-homogeneous right-hand side of the continuity condition (for which GMRes solver
    is not stable in this library).
    The initial data and RHS needs to be specified in an object exact. VTK output can be provided if enabled.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma and is (ideally) refined around it.
        dt: float
            Time step size.
        order: int
            Polynomial order for FEM, default 1.
        tfinal: float
            Final time in the simulation.
        exact: Exact
            Exact solution object, see laplace_solvers.py file.
        band: float
            Size of the band around levelset, the distance metric is in terms of the levelset function.
        time_order: int
            Order of time discretization (BDF in this case).
        out: bool
            Flag that indicates if VTK output is to be created.
        fname: str
            File name for VTK output.
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

    # MESH

    # Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(exact.phi)
    lset_approx = GridFunction(H1(mesh, order=1, dirichlet=[]))
    ba_IF = BitArray(mesh.ne)
    ba_IF_band = BitArray(mesh.ne)

    update_geometry(mesh, exact.phi, lset_approx, band, ba_IF, ba_IF_band)

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

    if out:
        gfu_out = GridFunction(V)
        gfp_out = GridFunction(Q)
        if fname:
            filename = f"./vtk_out/diffusion/moving-ns-{fname}"
        else:
            filename = "./vtk_out/diffusion/moving-ns"
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, exact.phi, gfu_out, exact.u, gfp_out, exact.p],
                        names=["P1-levelset", "phi", "u", "uSol", "p", "pSol"],
                        filename=filename,
                        subdivision=0)

    set_ic(
        mesh=mesh, V=V, order=order, gfu_prevs=gfu_prevs, exact=exact, dt=dt,
        lsetmeshadap=lsetmeshadap, lset_approx=lset_approx, band=band, ba_IF=ba_IF, ba_IF_band=ba_IF_band,
        n=n, Pmat=Pmat, rho_u=rho_u, tau=tau, ds=ds, dX=dX
    )

    dofs = []
    # TIME MARCHING
    exact.set_time(0.0)
    t_curr = 0.0

    if out:
        gfu_out.Set(gfu_prevs[0])
        gfp_out.Set(exact.p)
        vtk.Do(time=t_curr)

    i = 1

    l2err_old = 0.0

    while t_curr < tfinal + dt/2:
        exact.set_time(t_curr + dt)
        with TaskManager():
            deformation = lsetmeshadap.CalcDeformation(exact.phi)

            update_geometry(mesh, exact.phi, lset_approx, band, ba_IF, ba_IF_band)

            VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
            QG = Compress(Q, GetDofsOfElements(Q, ba_IF))
            XG = FESpace([VG, QG])
            dofs.append(XG.ndof)

            # helper grid functions
            n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=exact.phi, vel_space=V)

        gfu_approx = GridFunction(VG)
        if time_order == 1:
            gfu_approx.Set(Pmat * gfu_prevs[0])
        elif time_order == 2:
            gfu_approx.Set(2 * Pmat * gfu_prevs[0] - Pmat * gfu_prevs[1])
        else:
            gfu_approx.Set(3 * Pmat * gfu_prevs[0] - 3 * Pmat * gfu_prevs[1] + Pmat * gfu_prevs[2])

        a, f, = define_forms(
            VG=VG, QG=QG,
            bdf_coeff=bdf_coeff, dt=dt,
            n=n, Pmat=Pmat, n_k=n_k, Hmat=Hmat,
            tau=tau, rho_u=rho_u, rho_p=rho_p,
            ds=ds, dX=dX, dX2=dX2,
            exact=exact, gfu_approx=gfu_approx, gfu_prevs=gfu_prevs,
            solver='direct', XG=XG
        )

        with TaskManager():
            assemble_forms([a, f])
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

        if out:
            gfu_out.Set(gfu)
            gfp_out.Set(gfp)
            vtk.Do(time=t_curr+dt)

        l2p, h1p = errors_scal(mesh, ds, Pmat, gfp, exact.p)
        l2u, h1u = errors_vec(mesh, ds, Pmat, gfu, exact.u)

        gfu_prevs[0].Set(gfu)
        mass_append(keys=keys, vals=[t_curr, l2u, h1u, l2p, h1p], **out_errs)

        print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        l2err_old = l2err
        t_curr += dt
        i += 1
    print("")

    return np.mean(dofs), out_errs['ts'], out_errs['l2us'], out_errs['h1us'], out_errs['l2ps'], out_errs['h1ps']


def moving_ns(mesh, dt, order, tfinal, exact, band, time_order=2, out=False, fname=None):
    """
    Solves evolving-surface Tangential Navier-Stokes on a provided mesh using GMRes. Intended to be used with zero RHS
    of the continuity equation (i.e. g=0). Otherwise might be unstable.
    The initial data and RHS needs to be specified in an object exact. VTK output can be provided if enabled.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma and is (ideally) refined around it.
        dt: float
            Time step size.
        order: int
            Polynomial order for FEM, default 1.
        tfinal: float
            Final time in the simulation.
        exact: Exact
            Exact solution object, see laplace_solvers.py file.
        band: float
            Size of the band around levelset, the distance metric is in terms of the levelset function.
        time_order: int
            Order of time discretization (BDF in this case).
        out: bool
            Flag that indicates if VTK output is to be created.
        fname: str
            File name for VTK output.
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

    # Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(exact.phi)
    lset_approx = GridFunction(H1(mesh, order=1, dirichlet=[]))
    ba_IF = BitArray(mesh.ne)
    ba_IF_band = BitArray(mesh.ne)

    update_geometry(mesh, exact.phi, lset_approx, band, ba_IF, ba_IF_band)

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

    if out:
        gfu_out = GridFunction(V)
        gfp_out = GridFunction(Q)
        if fname:
            filename = f"./vtk_out/diffusion/moving-ns-{fname}"
        else:
            filename = "./vtk_out/diffusion/moving-ns"
        vtk = VTKOutput(mesh,
                        coefs=[lset_approx, exact.phi, gfu_out, exact.u, gfp_out, exact.p],
                        names=["P1-levelset", "phi", "u", "uSol", "p", "pSol"],
                        filename=filename,
                        subdivision=0)

    set_ic(
        mesh=mesh, V=V, order=order, gfu_prevs=gfu_prevs, exact=exact, dt=dt,
        lsetmeshadap=lsetmeshadap, lset_approx=lset_approx, band=band, ba_IF=ba_IF, ba_IF_band=ba_IF_band,
        n=n, Pmat=Pmat, rho_u=rho_u, tau=tau, ds=ds, dX=dX
    )

    dofs = []
    # TIME MARCHING
    exact.set_time(0.0)
    t_curr = 0.0

    if out:
        gfu_out.Set(gfu_prevs[0])
        gfp_out.Set(exact.p)
        vtk.Do(time=t_curr)

    i = 1

    while t_curr < tfinal + dt/2:
        exact.set_time(t_curr + dt)
        t_curr += dt
        with TaskManager():
            deformation = lsetmeshadap.CalcDeformation(exact.phi)

            update_geometry(mesh, exact.phi, lset_approx, band, ba_IF, ba_IF_band)

            VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
            QG = Compress(Q, GetDofsOfElements(Q, ba_IF))
            dofs.append(VG.ndof + QG.ndof)

            # helper grid functions
            n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=exact.phi, vel_space=V)

        gfu_approx = GridFunction(VG)
        if time_order == 1:
            gfu_approx.Set(Pmat * gfu_prevs[0])
        elif time_order == 2:
            gfu_approx.Set(2 * Pmat * gfu_prevs[0] - Pmat * gfu_prevs[1])
        else:
            gfu_approx.Set(3 * Pmat * gfu_prevs[0] - 3 * Pmat * gfu_prevs[1] + Pmat * gfu_prevs[2])

        a, ap, pd, sq, b, c, f, g = define_forms(
            VG=VG, QG=QG,
            bdf_coeff=bdf_coeff, dt=dt,
            n=n, Pmat=Pmat, n_k=n_k, Hmat=Hmat,
            tau=tau, rho_u=rho_u, rho_p=rho_p,
            ds=ds, dX=dX, dX2=dX2,
            exact=exact, gfu_approx=gfu_approx, gfu_prevs=gfu_prevs,
            solver='gmres'
        )

        with TaskManager():
            pre_a = Preconditioner(a, "bddc")
            pre_pd = Preconditioner(pd, "bddc")
            pre_sq = Preconditioner(sq, "bddc")

            assemble_forms([a, ap, pd, sq, b, c, f, g])

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

            l2p, h1p = errors_scal(mesh, ds, Pmat, gfp, exact.p)
            l2u, h1u = errors_vec(mesh, ds, Pmat, gfu, exact.u)

            gfu_prevs[0].Set(gfu)
            mass_append(keys=keys, vals=[t_curr, l2u, h1u, l2p, h1p], **out_errs)

        print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        i += 1
    print("")

    return np.mean(dofs), out_errs['ts'], out_errs['l2us'], out_errs['h1us'], out_errs['l2ps'], out_errs['h1ps']
