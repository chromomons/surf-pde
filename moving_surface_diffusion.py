# ------------------------------ LOAD LIBRARIES -------------------------------
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from xfem.lsetcurv import *
from ngsolve import solvers
from ngsolve import TaskManager
import time
import sys
import numpy as np
from utils import errors_scal, mass_append, update_ba_IF_band


class Exact:
    """
    A class for exact solution of time-dependent problems. Its usage is motivated by the fact that it can store
    its own state (time). Might become an interface in later versions.
    """
    def __init__(self, nu, R, maxvel):
        """
        Default constructor, which initializes time of the object.
        Args:
            nu: float
                Diffusion parameter.
            R: float
                Radius of the domain in terms of the distance function phi.
            maxvel: float
                Maximum norm of the velocity over space and simulation time.
        """
        self.t = Parameter(0.0)
        self.nu = Parameter(nu)
        self.R = Parameter(R)
        self.maxvel = maxvel

        self.phi = None
        self.w = None
        self.u = None
        self.f = None
        self.g = None
        self.fel = None
        self.divGw = None
        self.divGwT = None

    def set_params(self, phi, w, u, f, fel, divGw, divGwT):
        """
        Sets parameters of the exact solution
        Args:
            phi: CoefficientFunction
                Levelset function.
            w: Vector-valued CoefficientFunction
                Ambient velocity.
            u: CoefficientFunction
                Solution.
            f: CoefficientFunction
                RHS of the PDE.
            fel: CoefficientFunction
                RHS of the auxiliary problem for normal extension of the initial condition(s).
            divGw: CoefficientFunction
                Gamma-divergence of the ambient velocity field (needed for the bilinear form).
            divGwT: CoefficientFunction
                Gamma-divergence of the tangential component of the ambient velocity field (needed for the bilinear
                form).
        Returns:

        """
        self.phi = phi
        self.w = w
        self.u = u
        self.f = f
        self.fel = fel
        self.divGw = divGw
        self.divGwT = divGwT

    def set_time(self, tval):
        """
        Changes the time of the Exact solution object to tval.
        Args:
            tval: float
                New time of the exact solution object.
        Returns:

        """
        self.t.Set(tval)


def moving_diffusion(mesh, dt, order, tfinal, exact, time_order=1, out=False):
    """
    Solves evolving-surface diffusion equation on a provided mesh. The initial data and RHS needs to be specified in a
    dictionary exact. VTK output can be provided if enabled.
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
            Exact solution object, see moving_surface_diffusion.py file.
        time_order: int
            Order of time discretization (BDF in this case).
        out: bool
            Flag that indicates if VTK output is to be created.

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
    bdf_coeffs = [[1, -1],
                  [3 / 2, -2, 1 / 2],
                  [11 / 6, -3, 3 / 2, -1 / 3]]

    bdf_coeff = bdf_coeffs[time_order-1]

    c_delta = time_order + 0.1
    # MESH

    exact.set_time(0.0)

    V = H1(mesh, order=order, dirichlet=[])

    # LEVELSET ADAPTATION
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
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

    keys = ['ts', 'l2us', 'h1us']
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
            l2u, h1u = errors_scal(mesh, ds, Pmat, gfu_prevs[j], exact.u)
        mass_append(keys=keys, vals=[t_curr, l2u, h1u], **out_errs)

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
            l2u, h1u = errors_scal(mesh, ds, Pmat, gfu, exact.u)
        mass_append(keys=keys, vals=[t_curr, l2u, h1u], **out_errs)

        i += 1
    print("")

    return np.mean(dofs), out_errs['ts'], out_errs['l2us'], out_errs['h1us']
