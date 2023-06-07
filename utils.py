from xfem.lsetcurv import *
from netgen.csg import CSGeometry, OrthoBrick, Pnt
from ngsolve import TaskManager
from sympy.parsing.mathematica import mathematica
from sympy import printing
import matplotlib.pyplot as plt
import seaborn as sns

# FORMATTING TOOLS
class bcolors:
    """
    Class for printing in different colors.
    from https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    """
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
    """
    Prints string s in bold.
    Args:
        s: str
            String to be printed

    Returns:

    """
    print(f"{bcolors.BOLD}{s}{bcolors.ENDC}")
    return


# MESHING

def background_mesh(unif_ref=2, bbox_sz=4./3):
    """
    Creates coarse background mesh in the form of a cube of size 2*bbox_sz,
    and performs unif_ref uniform refinement steps.

    Args:
        unif_ref: int
            Number of uniform refinements
        bbox_sz: float
            Half of the box size

    Returns:
        mesh: ngsolve.comp.Mesh.Mesh
            Newly generated coarse background mesh
    """
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
    for i in range(unif_ref - 1):
        with TaskManager():
            mesh.Refine()
    return mesh


def refine_at_levelset(mesh, levelset, nref=1):
    """
    Performs nref refinements around levelset.

    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Input mesh
        levelset: CoefficientFunction
            The levelset function
        nref: int
            Number of refinements around the levelset
    Returns:

    """
    for i in range(nref):
        lsetp1 = GridFunction(H1(mesh, order=1))
        InterpolateToP1(levelset, lsetp1)
        RefineAtLevelSet(lsetp1)
        with TaskManager():
            mesh.Refine()


def refine_around_lset(mesh, nref, phi, band, miniband, band_type='both'):
    """
    Refines on a band around levelset.
    The use case is to refine uniformly around levelset at time t=0
    in a region that contains the entire trajectory of the surface motion.

    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Input mesh
        nref: int
            Number of refinements on a around lset
        phi: CoefficientFunction
            The levelset function
        band: float
            The size of the band in the direction of motion
        miniband: float
            The size of the band in the direction where there is no motion
            In particular, this is nonzero for BDFk when k > 1
        band_type: str
            If the surface motion is only inward, prevents refining outside,
            and vice versa. Three possible values: 'inner', 'outer' and 'both' (default)
    Returns:

    """
    for i in range(nref):
        lset_p1 = GridFunction(H1(mesh=mesh, order=1, dirichlet=[]))
        InterpolateToP1(phi, lset_p1)
        if band_type == 'outer':
            RefineAtLevelSet(lset_p1, lower=-miniband, upper=band)
        elif band_type == 'inner':
            RefineAtLevelSet(lset_p1, lower=-band, upper=miniband)
        else:
            RefineAtLevelSet(lset_p1, lower=-band, upper=band)
        with TaskManager():
            mesh.Refine()


# DIFFERENTIAL OPERATORS

def coef_fun_grad(u):
    """
    Computes gradient of a scalar coefficient function
    Args:
        u: CoefficientFunction

    Returns:

    """
    return CoefficientFunction(tuple([u.Diff(d) for d in [x, y, z]]))


def vec_grad(v):
    """
    Computes gradient of a (column) vector-valued Coefficient function (3d).
    Args:
        v: vector-valued Coefficient function (3d)

    Returns:
        A tensor-valued gradient Coefficient function
    """
    return CoefficientFunction(tuple([v[i].Diff(d) for i in [0, 1, 2] for d in [x, y, z]]), dims=(3, 3))


# CONVENIENCE FUNCTIONS

def assemble_forms(list_of_forms):
    """
    Assembles each form in the list
    """
    for form in list_of_forms:
        form.Assemble()


def mass_append(keys, vals, **dict):
    """
    Given a list of keys, and a list of values, appends each val into dict[key] in order
    Args:
        keys: List
            A list of keys to append values to
        vals: List
            A list of values to append
        **dict: Dictionary
            A dictionary to append values to

    Returns:

    """
    for key, val in zip(keys, vals):
        dict[key].append(val)


def print_test_info(problem_name):
    """
    Prints test information based on the problem.
    Args:
        problem_name: str
            Name of the problem
    Returns:

    """
    name_dict = {'fixed_surface_poisson': "fixed-surface Poisson",
                 'fixed_surface_diffusion': "fixed-surface diffusion",
                 'evolving_surface_diffusion': "evolving-surface diffusion",
                 'fixed_surface_steady_stokes': "fixed-surface steady Stokes",
                 'fixed_surface_unsteady_stokes': "fixed-surface unsteady Stokes",
                 'fixed_surface_navier_stokes': "fixed-surface Navier-Stokes",
                 'evolving_surface_navier_stokes': "evolving-surface Navier-Stokes"}
    printbf(f"Solving {name_dict[problem_name]} problem.")


# ERRORS

def errors_scal(mesh, ds, Pmat, scal_gf, scal_cf):
    """
    Computes L^2 and H^1 norms between scal_gf and scal_cf over surface Gamma
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma
        ds: xfem.dCul
            Element of area on Gamma
        Pmat: Tensor-valued GridFunction
            Approximate projection matrix
        scal_gf: GridFunction
            Scalar grid function conforming to mesh
        scal_cf: scalar CoefficientFunction
            Intended to be the exact solution

    Returns:
        float, float
            L^2 and H^1 errors over Gamma
    """
    return sqrt(Integrate(InnerProduct(scal_gf - scal_cf, scal_gf - scal_cf) * ds, mesh=mesh)),\
           sqrt(Integrate(InnerProduct(Pmat * (grad(scal_gf) - coef_fun_grad(scal_cf)), Pmat * (grad(scal_gf) - coef_fun_grad(scal_cf))) * ds, mesh=mesh))


def errors_vec(mesh, ds, Pmat, vec_gf, vec_cf):
    """
        Same as errors_scal, but for vector-valued functions.
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma
        ds: xfem.dCul
            Element of area on Gamma
        Pmat: Tensor-valued GridFunction
            Approximate projection matrix
        scal_gf: Vector-valued GridFunction
            Vector-valued grid function conforming to mesh
        scal_cf: Vector-valued CoefficientFunction
            Intended to be the exact solution

    Returns:
        float, float
            L^2 and H^1 errors over Gamma
    """
    return sqrt(Integrate(InnerProduct(vec_gf - vec_cf, Pmat * (vec_gf - vec_cf)) * ds, mesh=mesh)),\
           sqrt(Integrate(InnerProduct((grad(vec_gf) - vec_grad(vec_cf)) * Pmat, Pmat * (grad(vec_gf) - vec_grad(vec_cf)) * Pmat) * ds, mesh=mesh))


# STOKES HELPERS

def get_dom_measure(Q, mesh, ds):
    """
    Computes the area of the discrete surface
    Args:
        Q: H1
            Scalar-valued FE space defined on Gamma_h
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma_h
        ds: xfem.dCul
            Element of area on Gamma

    Returns:
        float
            area of the discrete surface

    """
    one = GridFunction(Q)
    one.Set(CoefficientFunction(1.0))
    domMeas = Integrate(one * ds, mesh)
    return domMeas


def renormalize(Q, mesh, ds, gfp, domMeas=None):
    """
    Given scalar GridFunction gfp, subtracts its mean from it in order to make it mean zero over Gamma_h
    Args:
        Q: H1
            Scalar-valued FE space on which gfp is defined
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma_h
        ds: xfem.dCul
            Element of area on Gamma
        gfp: GridFunction(Q)
            GridFunction on Q, which we want to renormalize
        domMeas: float
            Area of Gamma_h

    Returns:

    """
    gfpInt = Integrate(gfp * ds, mesh)
    if not domMeas:
        domMeas = get_dom_measure(Q, mesh, ds)
    gfpMeanVal = GridFunction(Q)
    gfpMeanVal.Set(CoefficientFunction(float(gfpInt/domMeas)))
    pNum = GridFunction(Q)
    pNum.Set(gfp - gfpMeanVal)
    gfp.Set(pNum)


def helper_grid_functions(mesh, order, levelset, vel_space):
    """
    For [P_k]^3 approximation of the velocity, returns a k-th order approximation of
    the unit normal, and (k-2)-nd order approximation of the shape operator. These
    higher-order approximations are necessary for higher-order consistency estimates
    for Stokes and Navier-Stokes, see Brandner et al, http://arxiv.org/abs/2103.03843
    Args:
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma_h
        order: int
            Polynomial order of the velocity field
        levelset: CoefficientFunction
            Levelset function
        vel_space: Restriction of H1(order=order) to bulk around discrete levelset

    Returns:
        n_k: Vector-valued GridFunction
            k-th order approximation of the unit normal.
        Hmat: Tensor-valued GridFunction
            (k-2)-nd order approximation of the shape operator.

    """
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


# MOVING DOMAINS

def update_ba_IF_band(lset_approx, mesh, band_size, ba_IF_band):
    """
    Updates BitArray of elements in the narrow band of size band_size around the discrete levelset function
    (lset_approx) on the provided mesh. Intended to be used with evolving domains, where active dofs are changing
    in time.
    Args:
        lset_approx: GridFunction
            P1-approximation of the levelset function
        mesh: ngsolve.comp.Mesh.Mesh
            Mesh that contains surface Gamma_h
        band_size: float
            Size of the narrowband, in terms of the distance function lset_approx (not necessarily Euclidean distance)
        ba_IF_band: pyngcore.BitArray
            BitArray of active elements to be updated

    Returns:

    """
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


# PARSING MATHEMATICA INPUT
def sympy_to_cf(func, params):
    """
    Converts a sympy expression to a NGSolve coefficient function. Taken from
    https://ngsolve.org/forum/ngspy-forum/746-gradient-of-finite-element-function-inaccurately-integrated
    Args:
        func: str
            Mathematica expression as a string.
        params: dict
            A dictionary of parameters of the expression other than x,y,z,t.

    Returns:
        out: CoefficientFunction
            A coefficient function corresponding to the mathematica expression.
    """
    out = {}
    exec('from ngsolve import *; cf='+printing.sstr(func)+';', params, out)
    return out['cf']


def math_dict_to_cfs(d, params):
    """
    Takes dictionary d as an input and returns a dictionary of coefficient functions with the same keys.
    Args:
        d: dict
            A dictionary of Mathematica expressions.
        params: dict
            A dictionary of parameters of the expression other than x,y,z,t.
    Returns:
        cfs: dict
            A dictionary of corresponding coefficient functions with the same keys.
    """
    cfs = {}
    for key, value in d.items():
        cfs[key] = sympy_to_cf(mathematica(value), params)
    return cfs


# Generate error plots
def plt_out(ts, l2us, h1us, title, fname, l2ps=None, h1ps=None):
    """
    Produces error vs time plots.
    Args:
        ts: List[float]
            List of times t_n at which FEM solution was computed.
        l2us: List[float]
            Errors in L^2 norm over Gamma for u corresponding to discrete times in ts.
        h1us: List[float]
            Errors in H^1 norm over Gamma for u corresponding to discrete times in ts.
        title: str
            Title of the plot.
        fname: str
            Filename for the figure.
        l2ps: List[float], default: None
            If provided, errors in L^2 norm over Gamma for p corresponding to discrete times in ts.
        h1ps: List[float], default: None
            If provided, errors in H^1 norm over Gamma for p corresponding to discrete times in ts.
    Returns:

    """
    sns.set()
    if l2ps and h1ps:
        fig, axs = plt.subplots(2, 2)
        latex_u = r"$\mathbf{{u}}_T$"
    else:
        fig, axs = plt.subplots(1, 2)
        latex_u = "$u$"

    fig.set_figwidth(20)
    fig.set_figheight(15)

    plt.suptitle(rf"{title}")

    axs[0, 0].plot(ts, l2us)
    axs[0, 0].set_title(rf"$L^2$-error in {latex_u}")
    axs[0, 0].set_ylim(0.0)

    axs[0, 1].plot(ts, h1us)
    axs[0, 1].set_title(rf"$H^1$-error in {latex_u}")
    axs[0, 1].set_ylim(0.0)

    if l2ps and h1ps:
        axs[1, 0].plot(ts, l2ps)
        axs[1, 0].set_title(rf"$L^2$-error in $p$")
        axs[1, 0].set_ylim(0.0)

        axs[1, 1].plot(ts, h1ps)
        axs[1, 1].set_title(rf"$H^1$-error in $p$")
        axs[1, 1].set_ylim(0.0)

    plt.show()
    plt.savefig(f"./output/plt_out/{fname}.png")
