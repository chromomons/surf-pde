from xfem.lsetcurv import *
from netgen.csg import CSGeometry, OrthoBrick, Pnt
from ngsolve import TaskManager


def background_mesh(unif_ref=2, bbox_sz=4./3):
    geo = CSGeometry()
    geo.Add(OrthoBrick(Pnt(-bbox_sz, -bbox_sz, -bbox_sz), Pnt(bbox_sz, bbox_sz, bbox_sz)))
    mesh = Mesh(geo.GenerateMesh(maxh=2*bbox_sz, quad_dominated=False))
    for i in range(unif_ref):
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
