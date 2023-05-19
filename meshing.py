from xfem.lsetcurv import *
from netgen.csg import CSGeometry, OrthoBrick, Pnt
from ngsolve import TaskManager


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
