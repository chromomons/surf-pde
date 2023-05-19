from xfem.lsetcurv import *
from netgen.csg import CSGeometry, OrthoBrick, Pnt
from ngsolve import TaskManager


def background_mesh(unif_ref=2, bbox_sz=4./3):
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
    for i in range(nref):
        lsetp1 = GridFunction(H1(mesh, order=1))
        InterpolateToP1(levelset, lsetp1)
        RefineAtLevelSet(lsetp1)
        with TaskManager():
            mesh.Refine()


def refine_around_lset(mesh, unif_ref, phi, band, miniband, band_type='both'):
    for i in range(unif_ref - 2):
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
    return