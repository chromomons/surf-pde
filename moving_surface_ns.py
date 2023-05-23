# ------------------------------ LOAD LIBRARIES -------------------------------
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from xfem.lsetcurv import *
from ngsolve import solvers
from ngsolve import TaskManager
import time
import sys
import numpy as np
from utils import renormalize, errors_scal, errors_vec, mass_append, background_mesh, refine_around_lset, \
    update_ba_IF_band, helper_grid_functions


# EXACT SOLUTION CLASS

class Exact:
    def __init__(self, mu, R, maxvel):
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
        self.divGw = None
        self.divGwT = None

    def set_params(self, phi, wN, u, p, f, fel, g):
        self.phi = phi
        self.wN = wN
        self.u = u
        self.p = p
        self.f = f
        self.fel = fel
        self.g = g

    def set_time(self, tval):
        self.t.Set(tval)


# SOLVERS

def moving_ns_direct(order, unif_ref, bbox_sz, tfinal, time_order=1, mu=1.0, out=False, fname=None, test_name="advect"):
    bdf_coeffs = [[1, -1],
                  [3 / 2, -2, 1 / 2],
                  [11 / 6, -3, 3 / 2, -1 / 3]]

    bdf_coeff = bdf_coeffs[time_order - 1]

    c_delta = (time_order + 0.5)
    # MESH
    mesh = background_mesh(bbox_sz=bbox_sz)

    h_approx = bbox_sz * 2 ** (1 - unif_ref)

    t = Parameter(0.0)

    dt = h_approx ** (order / time_order) / 4

    if test_name == "advect":
        vel = 0.2
        w1 = CoefficientFunction(vel)
        w2 = CoefficientFunction(0.0)
        w3 = CoefficientFunction(0.0)
        w = CoefficientFunction((w1, w2, w3))

        R = 1.0
        phi = -R + sqrt((-t*w1 + x)**2 + (-t*w2 + y)**2 + (-t*w3 + z)**2)
    else:
        vel = 0.2
        w1 = CoefficientFunction(vel)
        w2 = CoefficientFunction(0.0)
        w3 = CoefficientFunction(0.0)
        w = CoefficientFunction((w1, w2, w3))

        R = 1.0
        phi = -R + sqrt((-t * w1 + x) ** 2 + (-t * w2 + y) ** 2 + (-t * w3 + z) ** 2)

    c_delta = time_order + 0.1
    band = vel * (tfinal + c_delta * dt)
    miniband = c_delta * dt * vel
    refine_around_lset(mesh, unif_ref-2, phi, band, miniband, band_type="both")

    if test_name == "advect":
        wN = (-t*(w1**2 + w2**2 + w3**2) + w1*x + w2*y + w3*z)/sqrt(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)

        coef_u = CoefficientFunction((
                x * (t * w3 - z) / sqrt(
            t ** 2 * (w1 ** 2 + w2 ** 2 + w3 ** 2) - 2 * t * (w1 * x + w2 * y + w3 * z) + x ** 2 + y ** 2 + z ** 2),
                y * (-t * w3 + z) / sqrt(
                    t ** 2 * (w1 ** 2 + w2 ** 2 + w3 ** 2) - 2 * t * (w1 * x + w2 * y + w3 * z) + x ** 2 + y ** 2 + z ** 2),
                (-t * w1 * x + t * w2 * y + x ** 2 - y ** 2) / sqrt(
                    t ** 2 * (w1 ** 2 + w2 ** 2 + w3 ** 2) - 2 * t * (w1 * x + w2 * y + w3 * z) + x ** 2 + y ** 2 + z ** 2)
        ))
        coef_p = y*(-t*w1 + x) + z

        coef_f = CoefficientFunction((
            -4*mu*(t*w1 - x)*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2) + x + y + (t*w1 - x)*(t*w1*y + t*w2*x - 2*x*y)**2/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t*w1 - x)*(t**2*w1*w2 - t*(2*w1*y + w2*x + w3) + 2*x*y + z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (t*w2*x*y - x*(t**2*(w1**2 + w2**2) - 2*t*w1*x + x**2) + y**2*(-t*w1 + x))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (-t**2*w1*z*(w1**2 + w2**2 + 2*w3**2) + t*w2*w3*y*(-t*w1 + x) + t*w3*(t*w1 - 2*x)*(t*(w1**2 + w2**2 + w3**2) - w1*x) + t*z*(3*w1**2*x + 2*w1*w2*y + 3*w2**2*x + 4*w3**2*x) + w3*y**2*(-t*w1 + 2*x) + w3*z**2*(t*w1 - 2*x) - 2*x*z*(w1*x + 2*w2*y))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            4*mu*(t*w2 - y)*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2) - t*w1 + x + y + (t*w2 - y)*(t*w1*y + t*w2*x - 2*x*y)**2/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t*w2 - y)*(t**2*w1*w2 - t*(2*w1*y + w2*x + w3) + 2*x*y + z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (-t**2*y*(w1**2 + w2**2) + t*(w1*x*y - w2*x**2 + 2*w2*y**2) + y*(x - y)*(x + y))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (-t**3*w2*w3*(w1**2 + w2**2 + w3**2) + t**2*(w2*z*(w1**2 + w2**2 + 2*w3**2) + w3*(2*w1**2*y + w1*w2*x + 3*w2**2*y + 2*w3**2*y)) - t*(2*w1*w2*x*z + 3*w2**2*y*z + w2*w3*(-x**2 + 2*y**2 + z**2) + y*(3*w1**2*z + w1*w3*x + 4*w3**2*z)) + 2*y*(2*w1*x*z + w2*y*z - w3*x**2 + w3*z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            (4*mu*(t*(w1 - w2) - x + y)*(t*(w1 + w2) - x - y)*sqrt(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (t*w3 - z)*(-t**3*(w1*x + w2*y)*(w1**2 + w2**2 + w3**2) + t**2*(w1**2*(3*x**2 + 2*y**2) + 2*w1*x*(3*w2*y + w3*z) + w2**2*(2*x**2 + 3*y**2) + 2*w2*w3*y*z + w3**2*(x**2 + y**2)) - t*(w1*x*(3*x**2 + 7*y**2 + z**2) + w2*y*(7*x**2 + 3*y**2 + z**2) + 2*w3*z*(x**2 + y**2)) + x**4 + x**2*(6*y**2 + z**2) + y**2*(y**2 + z**2)) + sqrt(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)*(-t**3*(w1 - w2)*(w1 + w2)*(w1**2 + w2**2 + w3**2) + t**2*(4*w1**3*x + w1**2*(-2*w2*y + w3*z) + w1*x*(2*w2**2 + 3*w3**2) - w2*(4*w2**2*y + w2*w3*z + 3*w3**2*y)) + t*(w1**2*(-5*x**2 + 3*y**2) - 3*w1*w3*x*z + w2**2*(-3*x**2 + 5*y**2) + 3*w2*w3*y*z + 2*w3**2*(-x**2 + y**2)) + 2*w1*x**3 - 4*w1*x*y**2 + 4*w2*x**2*y - 2*w2*y**3 + 2*w3*z*(x - y)*(x + y)) + (t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)*(t**3*w1*w2*w3 + t**2*(w1*(w1 - w2*z - 2*w3*y) + w2*(w2 - w3*x)) + t*(-2*w1*x + 2*w1*y*z + w2*x*z - 2*w2*y + 2*w3*x*y) + x**2 - 2*x*y*z + y**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2
        ))

        coef_f_el = CoefficientFunction((
            (4*mu*(-t*w1 + x) + x*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            -(4*mu*(-t*w2 + y) + y*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))*(t*w3 - z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2),
            (4*mu*(t*(w1 - w2) - x + y)*(t*(w1 + w2) - x - y) - (t*w1*x - t*w2*y - x**2 + y**2)*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**(3/2)
        ))

        coef_g = CoefficientFunction(0.0)
    else:
        wN = (-t*(w1**2 + w2**2 + w3**2) + w1*x + w2*y + w3*z)/sqrt(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)
        coef_u = CoefficientFunction((
            -(-2*t*w3*z**3 + x*z*(-t*w1 + x) + z**4 + z**2*(t**2*(w2**2 + w3**2) - 2*t*w2*y + y**2) + (t*w1 - x)*(t*w2*y + t*w3*x - y**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2),
            (t**2*(w2*(w1*z**2 - w3*x) + y*(w1**2 + w3**2)) - t*(w1*y*(2*x + z**2) + w2*x*z*(z - 1) - w3*y*(x - 2*z)) + y*(x**2 + x*z*(z - 1) + z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2),
            (t**2*(w1**2*x + w1*w3*z**2 + w2*(w2*x - w3*y)) + t*(-w1*(2*x**2 + z**3) + w2*y*(-2*x + z) + w3*(-x*z**2 + y**2)) + x**3 + x*(y**2 + z**3) - y**2*z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)
        ))

        coef_p = CoefficientFunction(y**3*(-t*w1 + x) + z)

        coef_f = CoefficientFunction((
            mu*(2*t**4*(w1**2*(w2**2 + 2*w3**2) + w2**2*(w2**2 + w3**2)) - t**3*(w1**2*(4*w2*y + 3*w3*(6*z - 1)) + 4*w1*(w2**2*(x - 2) + 2*w3**2*x) + 8*w2**3*y + w2**2*w3*(5 - 2*z) + 4*w2*w3**2*y + w3**3*(5 - 6*z)) + t**2*(w1**2*(2*y**2 + z*(14*z - 3)) + 4*w1*w3*x*(9*z - 2) + w2**2*(x*(2*x - 8) + 12*y**2 + z*(5 - 6*z)) + 2*w2*y*(w1*(4*x - 9) + w3*(5 - 2*z)) + w3**2*(4*x**2 + 2*y**2 + z*(15 - 20*z))) + t*(2*w1*(2*x*z*(2 - 7*z) + y**2*(5 - 2*x)) - 2*w2*y*(x*(2*x - 9) + 4*y**2 + z*(5 - 6*z)) + w3*(x**2*(5 - 18*z) + y**2*(2*z - 5) + z**2*(22*z - 15))) + x**2*(2*y**2 + z*(14*z - 5)) - 10*x*y**2 + (y**2 + z**2)*(2*y**2 + z*(5 - 8*z)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + y**3 + (t*w1 - x)*(3*t*w2*y**2*(t*w1 - x) - t*w3 + y**3*(-4*t*w1 + 4*x) + z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (t*(w1**2 + w2**2 + w3**2) - w1*x - w2*y - w3*z)*(-2*t*w3*z**3 + x*z*(-t*w1 + x) + z**4 + z**2*(t**2*(w2**2 + w3**2) - 2*t*w2*y + y**2) + (t*w1 - x)*(t*w2*y + t*w3*x - y**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (-t**4*(w1*(w1*w3 + w2**2) + 2*w3*z*(w2**2 + w3**2))*(w1**2 + w2**2 + w3**2) + t**3*(w1**4*z + w1**3*(2*w2*y + 3*w3*x) + w1**2*(w2**2*(2*x + 2*z**2 + z) + w2*w3*y*(4*z + 1) + 2*w3**2*z*(3*z + 1)) + w1*(3*w2**3*y + w2**2*w3*(2*x*(z + 1) + z) + 2*w2*w3**2*y + 2*w3**3*x*(z + 1)) + (w2**2 + w3**2)*(w2**2*(x + 2*z**2) + 6*w2*w3*y*z + 8*w3**2*z**2)) - t**2*(w1**3*(3*x*z + y**2) + w1**2*(w2*y*(4*x + 5*z**2 + z) + 3*w3*x**2 + w3*z*(2*y**2 + 7*z**2 + z)) + w1*(w2**2*(x**2 + x*z*(z + 2) + 2*y**2) + w2*w3*y*(4*x*z + x + z) + w3**2*(x*z*(5*z + 3) + y**2)) + w2**3*(4*x*y + 6*y*z**2) + w2**2*w3*(2*x**2 + x*z + 6*y**2*z + 8*z**3) + w2*w3**2*y*(3*x + 14*z**2) + 2*w3**3*(x**2 + y**2*z + 6*z**3)) + t*(w1**2*(3*x**2*z + 2*x*y**2 + 3*z**2*(y**2 + z**2)) + w1*(-w2*y*(-2*x**2 - x*z*(4*z + 1) + y**2 + z**2) + w3*x*(x**2 + y**2*(2*z - 1) + 6*z**3)) + w2**2*(-x**2*z*(z - 2) + 5*x*y**2 + 2*z**2*(3*y**2 + z**2)) + w2*w3*y*(2*x**2 + 3*x*z + 2*z*(y**2 + 5*z**2)) + w3**2*(-x**2*z*(z - 4) + 2*x*y**2 + 4*z**2*(y**2 + 2*z**2))) + w1*(-x**3*z - x**2*y**2 - x*z*(y**2 + z**2)*(3*z - 1) + y**2*(y**2 + z**2)) - (w2*y + w3*z)*(-x**2*z*(z - 2) + 2*x*y**2 + 2*z**2*(y**2 + z**2)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t**4*(-w1**3*w2*y + w1**2*(-w2**2*z*(2*x + z) + w2*w3*y + w3**2*(-2*x*z + x)) + w1*w3*(w2**2*(x - 2*z**3 + z**2) - w2*w3*y + w3**2*z**2*(1 - 2*z)) - 2*w2*z*(w2**2 + w3**2)*(w2*x - w3*y)) + t**3*(w1**3*y**2 - w1**2*w3*(x*z*(2 - 5*z) + y**2) + w1*w3**2*(x**2*(4*z - 3) + y**2 + z**3*(7*z - 3)) + w2**3*y*z*(8*x - 3*z) - w2**2*(w1*y**2 + w1*z*(-4*x**2 - 2*x*z + x - 3*z**3 + z**2) + w3*x*(x - 2*z**2*(z + 1)) + 6*w3*y**2*z) + w2*y*(w1**2*(x*(4*z + 3) + z*(3*z - 1)) + 2*w1*w3*(-3*x + z**2*(2*z - 1) + z) + w3**2*(4*x*z + x - 7*z**2)) + 2*w3**3*z*(x*z*(z - 1) - y**2)) - t**2*(w1**2*(x*y**2*(2*z + 3) + x*z**2*(3*z - 1) + y**2*z*(2*z - 1)) + w1*w2*y*(x**2*(8*z + 3) + 6*x*z*(z - 1) - 2*y**2 + 6*z**4 - 2*z**3 + z**2) + w2**2*(2*x**3*z + x**2*z*(z - 1) + x*y**2*(12*z - 1) + 3*x*z**4 - 9*y**2*z**2) + w3**2*(2*x**3*(z - 1) + x*y**2*(2*z + 1) + x*z**3*(7*z - 6) - 7*y**2*z**2) + w3*(w1*(2*x**2*z*(5*z - 3) - 5*x*y**2 + y**2*z*(z*(2*z - 1) + 2) + z**4*(8*z - 3)) + w2*y*(-5*x**2 + 2*x*z*(2*z*(z + 1) + 1) - 6*y**2*z - 8*z**3))) + t*(w1*(x**2*(y**2*(4*z + 3) + 3*z**2*(2*z - 1)) + x*y**2*z*(4*z - 5) - y**4 + y**2*z**2*(z*(3*z - 1) + 1) + z**5*(3*z - 1)) + w2*y*(x**3*(4*z + 1) + x**2*z*(3*z - 5) + x*(y**2*(8*z - 2) + 6*z**4 + z**2) - 9*y**2*z**2 - 3*z**4) + w3*(x**3*z*(5*z - 4) - 4*x**2*y**2 + 2*x*z*(y**2*(z**2 + z + 1) + z**3*(4*z - 3)) - 2*y**2*z*(y**2 + 4*z**2))) - x**3*(y**2*(2*z + 1) + z**2*(3*z - 2)) - 2*x**2*y**2*z*(z - 2) + x*(y**4*(1 - 2*z) - y**2*(3*z**4 + z**2) + z**5*(2 - 3*z)) + 3*y**2*z**2*(y**2 + z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            -2*mu*(t**4*w1*w2*(w1**2 + w2**2 - w3**2) + t**3*(-w1**3*y + w1**2*w2*(4 - 3*x) + w1*(-3*w2**2*y + 2*w2*w3*(5*z - 2) + w3**2*y) - w2**3*x + w2*w3**2*(x + 4)) + t**2*(w1**2*y*(3*x - 5) + w1*w2*(x*(3*x - 8) + 3*y**2 + z*(4 - 10*z)) + 2*w1*w3*y*(2 - 5*z) + 3*w2**2*x*y + 5*w2*w3*x - 2*w2*w3*z*(5*x + 4) - w3**2*y*(x + 5)) + 2*t*z**2*(5*w1*y + w2*(5*x + 2)) + t*z*(-4*w1*y - 5*w2*x + 10*w3*y*(x + 1)) - t*(w1*y*(x*(3*x - 10) + y**2) + w2*x*(x*(x - 4) + 3*y**2) + 5*w3*x*y) + y*(5*x*z + x*(x*(x - 5) + y**2) - z**2*(10*x + 5)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + y**2*(-3*t*w1 + 3*x) + (t*w2 - y)*(3*t*w2*y**2*(t*w1 - x) - t*w3 + y**3*(-4*t*w1 + 4*x) + z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) - (t**2*(w2*(w1*z**2 - w3*x) + y*(w1**2 + w3**2)) - t*(w1*y*(2*x + z**2) + w2*x*z*(z - 1) - w3*y*(x - 2*z)) + y*(x**2 + x*z*(z - 1) + z**2))*(t*(w1**2 + w2**2 + w3**2) - w1*x - w2*y - w3*z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t**4*(w1**4*y + w1**3*w2*z*(2*x + z) + w1**2*w3*(-w2*x + 2*w2*z**3 + 2*w3*y) + w1*w2*(2*w2**2*x*z + w2*w3*y*(1 - 2*z) + w3**2*(x + z**2)) + w2**3*w3*z**2 + w2*w3**3*(-x + z**2) + w3**4*y) + t**3*(-2*w1**3*y*(x*(z + 2) + z**2) + w1**2*(w2*(y**2 + z*(-6*x**2 - 3*x*z + x - 3*z**3)) + 2*w3*y*(x - z*(z**2 + 2))) + w1*(w2**2*y*z*(-6*x + 3*z - 1) + 2*w2*w3*(x**2 + x*z*(-2*z**2 + z - 1) + y**2*(2*z - 1) - z**3) - w3**2*y*(5*x + 2*z**2)) - w2**3*(2*x**2*z + z**3) + w2**2*w3*y*(2*x*(z - 1) - 3*z**2) + w2*w3**2*(-2*x**2 - x*z*(z - 3) + y**2 - 3*z**3) + w3**3*y*(2*x - z*(z + 4))) + t**2*(w2**2*y*z*(6*x**2 + x*(2 - 3*z) + 3*z**2) + w2*(-w3*x**3 - x*y**2*(2*w1 - 4*w3) + 2*x*z**3*(-w1 + w3*x + w3) + z**4*(6*w1*x + w1 + 3*w3) + z**2*(x*(3*w1*x + w1 - w3*(2*x + 3)) + y**2*(-6*w1 + 3*w3)) + z*(2*x**2*(w1*(3*x - 1) + 2*w3) + 2*y**2*(3*w1*x + w1 - w3*(2*x + 1)))) + y*(w1**2*(6*x**2*(z + 1) + 2*x*z*(3*z - 1) - y**2 + 3*z**4 + 2*z**2) + w1*w3*(-4*x**2 + 2*x*z*(z*(2*z - 1) + 5) - 2*y**2*z + y**2 + 4*z**3) + w3**2*(4*x**2 + 2*x*z*(z - 3) - y**2 + 3*z**2*(z + 2)))) + t*(-w2*z**5 + x*y*(w2*x*y + x**2*(-4*w1 + 2*w3) + y**2*(2*w1 - 2*w3)) - z**4*(w1*y*(6*x + 2) + w2*x*(3*x + 1) + 3*w3*y) + z**3*(2*w1*x*y + w2*(2*x**2 + x - 3*y**2) - 2*w3*y*(x*(x + 2) + 2)) + z**2*(-w2*x**2*(x + 2) + w2*y**2*(6*x + 1) + x*y*(-w1*(6*x + 5) + 2*w3*(x + 3)) + y**3*(3*w1 - w3)) - z*(w2*x**3*(2*x - 1) + 2*w2*x*y**2*(3*x + 2) + 2*x**2*y*(w1*(3*x - 2) + 4*w3) + y**3*(2*w1*x + w1 - 2*w3*(x + 1)))) + y*(x**2*(x - y)*(x + y) + 2*x*z*(x**2*(x - 1) + y**2*(x + 1)) + z**5 + z**4*(x*(3*x + 2) + 1) + z**3*(-2*x*(x + 1) + y**2) + z**2*(2*x**2*(x + 2) - y**2*(3*x + 1))))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t**4*w2*(w1**2 + w2**2 + w3**2)*(w1**2 + w1*w3*(2*z - 1) + w3**2) + t**3*(-w1**4*y + w1**3*(w2*(-3*x - 2*z**2 + z) + w3*y*(1 - 2*z)) - w1**2*(-2*w2*w3*x + w2*w3*z*(4*x + 3) + y*(2*w2**2 + 2*w3**2)) + w1*(w2**3*(-2*x - 2*z**2 + z) + 2*w2**2*w3*y*(1 - 2*z) + w2*w3**2*(-3*x - 4*z**2 + 2*z) + w3**3*y*(1 - 2*z)) - w3*(w2**3*(-x + z*(2*x + 2)) + 2*w2**2*w3*y + w2*w3**2*(2*x*z - x + 3*z) + w3**3*y)) + t**2*(w1**3*y*(3*x + z*(3*z - 1)) + w1**2*(w2*(3*x**2 + x*z*(3*z - 2) + z**2) + w3*y*(x*(4*z - 3) + 3*z)) + w1*(w2**2*y*(5*x + 2*z*(2*z - 1)) + w2*w3*(y**2*(2*z - 1) + z*(2*x*(x + 2) + z*(z - 1))) + w3**2*y*(3*x + z*(5*z - 2))) + w2*z**2*(w2**2*(2*x + 1) + w3**2*(4*x + 3)) + x*(w2**2 + w3**2)*(w2*x - 2*w3*y) + z*(w2**2*w3*y*(4*x + 5) - w2*x*(w2**2 + w3**2) + w3**3*y*(2*x + 3))) + t*(w1**2*y*(-3*x**2 + 3*x*z + y**2 - z**2*(6*x + 1)) + w1*(-w2*(x**3 + x*(2*y**2 + z**2) - z*(-2*y**2*z + y**2 + z**3)) + w3*y*(2*x**2 - 2*x*z*(x + 2) - 3*z**3 + z**2)) - w2**2*y*(3*x**2 - 2*x*z + z**2*(4*x + 3)) - w2*w3*(x**3 + x**2*z + x*y**2*(2*z - 1) + x*z**2*(z + 1) + 2*y**2*z + z**3) + w3**2*y*(-x**2 + x*z*(4 - 5*z) + y**2 - 3*z**2)) + w1*x*y*(x**2 + x*z*(3*z - 2) - y**2 + z**2) + 2*w2*x**2*y**2 - w2*x*z**4 + z**3*(w2*x + w3*y*(3*x + 1)) + z**2*(-w2*x**3 + 2*w2*y**2*(x + 1) - 2*w3*x*y) + z*(x - y)*(x + y)*(w2*x + w3*y))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            mu*(-4*t**4*w1*w3*(w1**2 + w2**2) + t**3*(w1**3*(14*z - 3) + 12*w1**2*w3*x + w1*(w2**2*(14*z - 3) + 8*w2*w3*y + w3**2*(5 - 6*z)) + 4*w2**2*w3*(x + 2)) - t**2*(w1**2*x*(42*z - 11) + 2*w1*(w2*y*(14*z - 3) + w3*(6*x**2 + 2*y**2 + z*(5 - 7*z))) + w2**2*(14*x*z - 5*x + 8*z) + 2*w2*w3*y*(4*x + 9) + w3**2*x*(5 - 6*z)) + t*(w1*x**2*(42*z - 13) + w1*y**2*(14*z - 3) + w1*z**2*(5 - 8*z) + 2*w2*y*(x*(14*z - 5) + 9*z) + 2*w3*(2*x**3 + 2*x*y**2 + x*z*(5 - 7*z) + 5*y**2)) + 8*x*z**3 - 5*x*z**2 + 5*x*(x**2 + y**2) - z*(14*x**3 + 2*y**2*(7*x + 5)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 - (t*(w1**2 + w2**2 + w3**2) - w1*x - w2*y - w3*z)*(t**2*(w1**2*x + w1*w3*z**2 + w2*(w2*x - w3*y)) + t*(-w1*(2*x**2 + z**3) + w2*y*(-2*x + z) + w3*(-x*z**2 + y**2)) + x**3 + x*(y**2 + z**3) - y**2*z)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (3*t**3*w1*w2*w3*y**2 + t**2*(w1**2 - w1*y**2*(3*w2*z + 4*w3*y) + w2*(w2 - 3*w3*x*y**2)) + t*(-2*w1*(x - 2*y**3*z) + w2*y*(3*x*y*z - 2) + 4*w3*x*y**3) + x**2 - 4*x*y**3*z + y**2)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2) + (t**4*(-w1**3*(w2*y + w3*(-2*x*z + x)) - w1**2*(w2**2*z**2 + w2*w3*y + w3**2*z**2*(1 - 2*z)) - w1*w2*(w2**2*y + w2*w3*(-2*x*z + x + z**2) + 2*w3**2*y*z) + w2*(-w2*z**2*(w2**2 + w3**2) + w3**2*(w2*x - w3*y))) + t**3*(w1**3*(x*z*(1 - 3*z) + y**2) + w1**2*(w2*y*(4*x + 2*z**2 + z) + w3*(x**2*(4 - 6*z) + y**2 + z**3*(2 - 5*z))) + w1*(w2**2*(3*y**2 + z*(-x*z + x + z**2)) + 4*w2*w3*y*(-x*z + x + 2*z**2) + w3**2*z*(x*z*(3 - 4*z) + 2*y**2)) + 2*w2**3*y*(x + 2*z**2) + w2**2*w3*(-2*x**2*(z - 1) + x*z*(z - 2) - y**2 + 2*z**3) + w2*w3**2*y*(x*(2*z - 3) + z*(2*z + 3)) + w3**3*y**2) + t**2*(w1**2*(x**2*z*(9*z - 4) - 4*x*y**2 - y**2*z*(z + 1) + z**4*(3*z - 1)) + w1*(-w2*y*(5*x**2 - 2*x*z*(z - 2) + 3*y**2 + 6*z**3) + w3*(x**3*(6*z - 5) + x*y**2*(2*z - 3) + 2*x*z**3*(5*z - 3) - 7*y**2*z**2)) + w2**2*(2*x**2*z*(z - 1) + x*(-6*y**2 - z**3 + z**2) + y**2*z*(1 - 6*z) - z**4) + w2*w3*y*(x**2*(4*z - 5) + 2*x*z*(3 - 4*z) + 2*y**2 - z**2*(4*z + 3)) + w3**2*(2*x**2*z**2*(z - 1) - 2*x*y**2*(z - 1) - y**2*z*(z + 3))) + t*(w1*(x**3*z*(5 - 9*z) + 5*x**2*y**2 - x*y**2*z*(z - 3) + 3*x*z**4*(1 - 2*z) + y**4 + 5*y**2*z**3) + w2*y*(2*x**3 + x**2*z*(5 - 4*z) + 6*x*y**2 + 3*x*z**2*(2*z - 1) + y**2*z*(4*z - 2) + z**3*(2*z + 1)) + w3*(-2*x**4*(z - 1) + x**2*(y**2*(3 - 2*z) + z**3*(4 - 5*z)) + x*y**2*z*(7*z - 4) - y**4 + y**2*z**2*(2*z + 3))) + x**4*z*(3*z - 2) - 2*x**3*y**2 + x**2*z*(y**2*(2*z - 3) + z**3*(3*z - 2)) + x*y**2*(-2*y**2 + z**2*(2 - 5*z)) - y**2*z*(y**2*(z - 1) + z**2*(z + 1)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2 + (t**4*(w1**2 + w2**2 + w3**2)*(w1**3 + w1*(w2**2 + 2*w3**2*z) - w2**2*w3) + t**3*(-4*w1**4*x - w1**3*(3*w2*y + w3*z*(4*z + 1)) + w1**2*(w2**2*(-5*x + z) + 2*w2*w3*y - w3**2*x*(4*z + 3)) - w1*(3*w2**3*y + w2**2*w3*(-x + 4*z**2 + z) + 2*w2*w3**2*y*(z + 1) + 6*w3**3*z**2) + w2**4*(-x + z) + 3*w2**3*w3*y - w2**2*w3**2*(x + z*(2*x - 2)) + 2*w2*w3**3*y - 2*w3**4*x*z) + t**2*(w1**3*(6*x**2 + y**2 + 3*z**3) + w1**2*(w2*y*(7*x - 3*z) + w3*x*z*(7*z + 2) - w3*y**2) + w1*(w2**2*(4*x**2 - x*z + 3*y**2 + 3*z**3) + w2*w3*y*(-x + z*(3*z + 2)) + w3**2*(2*x**2*(z + 2) + y**2 + 6*z**3)) + w2**3*y*(3*x - 4*z) - w2**2*w3*(-4*x*z**2 + 2*y**2 + z**2) + w2*w3**2*y*(3*x + z*(2*x - 4)) - w3**3*(-6*x*z**2 + y**2)) + t*(w1**2*(-4*x**3 + x*(-2*y**2 - 6*z**3 + z**2) + 2*y**2*z) - w1*(w2*y*(5*x**2 - 3*x*z + y**2 + 3*z**3) + w3*z*(x**2*(2*z + 3) - y**2*z + y**2 + 2*z**3)) - w2**2*(x**3 + 3*x*y**2 + x*z**2*(3*z - 1) - 5*y**2*z) - w2*w3*y*(x**2 + x*z*(3*z + 2) + y**2 - 2*z**2) - 2*w3**2*(x**3 + x*(y**2 + 3*z**3) - y**2*z)) + 2*w3*x*z**4 + 3*x*z**3*(w1*x + w2*y) - z**2*(w2*x*y + w3*y**2*(x + 1) + x**2*(w1 + w3*x)) + z*(2*w3*x*(x**2 + y**2) - 2*y**2*(w1*x + w2*y)) + (x**2 + y**2)*(w1*x**2 + y*(w2*x + w3*y)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2
        ))

        coef_f_el = CoefficientFunction((
            (mu*(2*t**4*(w1**2*(w2**2 + 2*w3**2) + w2**2*(w2**2 + w3**2)) - t**3*(w1**2*(4*w2*y + 3*w3*(6*z - 1)) + 4*w1*(w2**2*(x - 2) + 2*w3**2*x) + 8*w2**3*y + w2**2*w3*(5 - 2*z) + 4*w2*w3**2*y + w3**3*(5 - 6*z)) + t**2*(w1**2*(2*y**2 + z*(14*z - 3)) + 4*w1*w3*x*(9*z - 2) + w2**2*(x*(2*x - 8) + 12*y**2 + z*(5 - 6*z)) + 2*w2*y*(w1*(4*x - 9) + w3*(5 - 2*z)) + w3**2*(4*x**2 + 2*y**2 + z*(15 - 20*z))) + t*(-4*w1*x*z*(7*z - 2) - 2*w1*y**2*(2*x - 5) - 2*w2*y*(x*(2*x - 9) + 4*y**2 + z*(5 - 6*z)) + w3*(x**2*(5 - 18*z) + y**2*(2*z - 5) + z**2*(22*z - 15))) + x**2*(2*y**2 + z*(14*z - 5)) - 10*x*y**2 + 2*y**4 - 6*y**2*z**2 + 5*y**2*z - 8*z**4 + 5*z**3) - (t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)*(-2*t*w3*z**3 + x*z*(-t*w1 + x) + z**4 + z**2*(t**2*(w2**2 + w3**2) - 2*t*w2*y + y**2) + (t*w1 - x)*(t*w2*y + t*w3*x - y**2)))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            (-2*mu*(t**4*w1*w2*(w1**2 + w2**2 - w3**2) + t**3*(-w1**3*y + w1**2*w2*(4 - 3*x) + w1*(-3*w2**2*y + 2*w2*w3*(5*z - 2) + w3**2*y) - w2**3*x + w2*w3**2*(x + 4)) + t**2*(w1**2*y*(3*x - 5) + w1*w2*(x*(3*x - 8) + 3*y**2 + z*(4 - 10*z)) + 2*w1*w3*y*(2 - 5*z) + 3*w2**2*x*y + w2*w3*(5*x - z*(10*x + 8)) - w3**2*y*(x + 5)) - t*(w1*y*(x*(3*x - 10) + y**2 + z*(4 - 10*z)) + 5*w2*x*z + w2*x*(x*(x - 4) + 3*y**2) - 2*w2*z**2*(5*x + 2) + 5*w3*y*(x - z*(2*x + 2))) + y*(5*x*z + x*(x*(x - 5) + y**2) - z**2*(10*x + 5))) + (t**2*(w2*(w1*z**2 - w3*x) + y*(w1**2 + w3**2)) - t*(w1*y*(2*x + z**2) + w2*x*z*(z - 1) - w3*y*(x - 2*z)) + y*(x**2 + x*z*(z - 1) + z**2))*(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2,
            (-mu*(4*t**4*w1*w3*(w1**2 + w2**2) - t**3*(w1**3*(14*z - 3) + 12*w1**2*w3*x + w1*(w2**2*(14*z - 3) + 8*w2*w3*y + w3**2*(5 - 6*z)) + 4*w2**2*w3*(x + 2)) + t**2*(w1**2*x*(42*z - 11) + 2*w1*(w2*y*(14*z - 3) + w3*(6*x**2 + 2*y**2 + z*(5 - 7*z))) + w2**2*(14*x*z - 5*x + 8*z) + 2*w2*w3*y*(4*x + 9) + w3**2*x*(5 - 6*z)) - t*(w1*x**2*(42*z - 13) + w1*y**2*(14*z - 3) + w1*z**2*(5 - 8*z) + 2*w2*y*(x*(14*z - 5) + 9*z) + 2*w3*(2*x**3 + 2*x*y**2 + x*z*(5 - 7*z) + 5*y**2)) + x**3*(14*z - 5) + x*y**2*(14*z - 5) + x*z**2*(5 - 8*z) + 10*y**2*z) + (t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)*(t**2*(w1**2*x + w1*w3*z**2 + w2*(w2*x - w3*y)) + t*(-w1*(2*x**2 + z**3) + w2*y*(-2*x + z) + w3*(-x*z**2 + y**2)) + x**3 + x*(y**2 + z**3) - y**2*z))/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2)**2
        ))

        coef_g = CoefficientFunction((t**2*(w1**2 + w1*w3*(2*z - 1) + w3**2) + t*(w1*(-2*x - 4*z**2 + z) + 2*w2*y + 3*w3*x - 2*w3*z*(x + 1)) + x**2 + x*z*(4*z - 3) - 2*y**2 + z**2)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))

    # Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(phi)
    lset_approx = GridFunction(H1(mesh, order=1, dirichlet=[]))
    InterpolateToP1(phi, lset_approx)
    ci = CutInfo(mesh, lset_approx)

    ba_IF = ci.GetElementsOfType(IF)
    ba_IF_band = BitArray(mesh.ne)
    update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

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
                        coefs=[lset_approx, phi, gfu_out, coef_u, gfp_out, coef_p],
                        names=["P1-levelset", "phi", "u", "uSol", "p", "pSol"],
                        filename=filename,
                        subdivision=0)

    for j in range(time_order):
        # fix levelset
        t.Set(-j * dt)
        t_curr = -j * dt

        deformation = lsetmeshadap.CalcDeformation(phi)

        # solve elliptic problem on a fixed surface to get u with normal extension
        InterpolateToP1(phi, lset_approx)
        ci = CutInfo(mesh, lset_approx)

        ba_IF.Clear()
        ba_IF |= ci.GetElementsOfType(IF)
        update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

        VG = Compress(V, GetDofsOfElements(V, ba_IF_band))

        # helper grid functions
        n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=phi, vel_space=V)

        gfu_el = GridFunction(VG)

        u_el, v_el = VG.TnT()

        a_el = BilinearForm(VG, symmetric=True)
        a_el += InnerProduct(Pmat * u_el, Pmat * v_el) * ds
        a_el += 2 * mu * (InnerProduct(Pmat * Sym(grad(u_el)) * Pmat - (u_el * n) * Hmat,
                                       Pmat * Sym(grad(v_el)) * Pmat - (v_el * n) * Hmat)) * ds
        a_el += (tau * ((u_el * n_k) * (v_el * n_k))) * ds
        a_el += (rho_u * InnerProduct(grad(u_el) * n, grad(v_el) * n)) * dX

        f_el = LinearForm(VG)
        f_el += InnerProduct(coef_f_el, Pmat * v_el) * ds

        with TaskManager():
            pre_a_el = Preconditioner(a_el, "bddc")

            a_el.Assemble()
            f_el.Assemble()

            solvers.CG(mat=a_el.mat, rhs=f_el.vec, pre=pre_a_el.mat, sol=gfu_el.vec, printrates=False)
            sys.stdout.write("\033[F\033[K")  # to remove annoying deprecation warning

            gfu_prevs[j].Set(gfu_el)

        with TaskManager():
            l2u, h1u = errors_vec(mesh, ds, Pmat, gfu_prevs[j], coef_u)
        mass_append(keys=keys, vals=[t_curr, l2u, h1u, 0, 0], **out_errs)

    # TIME MARCHING
    t.Set(0.0)
    t_curr = 0.0

    if out:
        gfu_out.Set(gfu_prevs[0])
        gfp_out.Set(coef_p)
        vtk.Do(time=t_curr)

    i = 1

    l2err_old = 0.0

    while t_curr < tfinal - dt:
        t.Set(t_curr + dt)
        with TaskManager():
            deformation = lsetmeshadap.CalcDeformation(phi)

            InterpolateToP1(phi, lset_approx)
            ci = CutInfo(mesh, lset_approx)

            ba_IF.Clear()
            ba_IF |= ci.GetElementsOfType(IF)
            update_ba_IF_band(lset_approx, mesh, c_delta * dt * vel, ba_IF_band)

            VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
            QG = Compress(Q, GetDofsOfElements(Q, ba_IF))
            XG = FESpace([VG, QG])
            u, p = XG.TrialFunction()
            v, q = XG.TestFunction()

            # helper grid functions
            n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=phi, vel_space=V)

        gfu_approx = GridFunction(VG)
        if time_order == 1:
            gfu_approx.Set(Pmat * gfu_prevs[0])
        elif time_order == 2:
            gfu_approx.Set(2 * Pmat * gfu_prevs[0] - Pmat * gfu_prevs[1])
        else:
            gfu_approx.Set(3 * Pmat * gfu_prevs[0] - 3 * Pmat * gfu_prevs[1] + Pmat * gfu_prevs[2])

        a = BilinearForm(XG, symmetric=False)
        a += bdf_coeff[0] / dt * InnerProduct(u, Pmat * v) * ds
        a += wN * InnerProduct(Hmat * u, Pmat * v) * ds
        a += 0.5 * InnerProduct((Pmat * grad(u) * Pmat - (u * n) * Hmat) * gfu_approx, v) * ds
        a += (-0.5) * InnerProduct((Pmat * grad(v) * Pmat - (v * n) * Hmat) * gfu_approx, u) * ds
        a += (-0.5) * InnerProduct(coef_g * u, Pmat * v) * ds
        a += 2.0 * mu * (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                                      Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        a += tau * InnerProduct(n_k, u) * InnerProduct(n_k, v) * ds
        a += rho_u * InnerProduct(grad(u) * n, grad(v) * n) * dX

        a += InnerProduct(u, Pmat * grad(q)) * ds
        a += InnerProduct(v, Pmat * grad(p)) * ds

        a += (-1.0) * rho_p * (grad(p) * n) * (grad(q) * n) * dX2

        f = LinearForm(XG)
        f += InnerProduct(coef_f, Pmat * v) * ds
        for j in range(time_order):
            f += (-1.0) * bdf_coeff[j + 1] / dt * InnerProduct(gfu_prevs[j], Pmat * v) * ds
        f += (-1.0) * coef_g * q * ds

        with TaskManager():
            a.Assemble()
            f.Assemble()

            gf = GridFunction(XG)

        gf.vec.data = a.mat.Inverse(freedofs=XG.FreeDofs(), inverse="umfpack") * f.vec

        gfu = gf.components[0]

        l2err = sqrt(Integrate(InnerProduct(Pmat * (gfu - gfu_prevs[0]), Pmat * (gfu - gfu_prevs[0])) * ds, mesh))

        if i > 1 and l2err > 5 * l2err_old:
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

        l2p, h1p = errors_scal(mesh, ds, Pmat, gfp, coef_p)
        l2u, h1u = errors_scal(mesh, ds, Pmat, gfu, coef_u)

        gfu_prevs[0].Set(gfu)
        mass_append(keys=keys, vals=[t_curr, l2u, h1u, l2p, h1p], **out_errs)

        print("\r", f"Time in the simulation: {t_curr:.5f} s ({int(t_curr / tfinal * 100):3d} %)", end="")

        l2err_old = l2err
        t_curr += dt
        i += 1

    return h_approx, dt, out_errs['ts'], out_errs['l2us'], out_errs['h1us'], out_errs['l2ps'], out_errs['h1ps']


def moving_ns(mesh, dt, order, tfinal, exact, time_order=2, out=False, fname=None):
    bdf_coeffs = [[1, -1],
                  [3 / 2, -2, 1 / 2],
                  [11 / 6, -3, 3 / 2, -1 / 3]]

    bdf_coeff = bdf_coeffs[time_order - 1]

    c_delta = time_order + 0.1

    # Levelset adaptation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order+1, heapsize=100000000)
    deformation = lsetmeshadap.CalcDeformation(exact.phi)
    lset_approx = GridFunction(H1(mesh, order=1, dirichlet=[]))
    InterpolateToP1(exact.phi, lset_approx)
    ci = CutInfo(mesh, lset_approx)

    ba_IF = ci.GetElementsOfType(IF)
    ba_IF_band = BitArray(mesh.ne)
    update_ba_IF_band(lset_approx, mesh, c_delta * dt * exact.maxvel, ba_IF_band)

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

    for j in range(time_order):
        # fix levelset
        exact.set_time(-j * dt)
        t_curr = -j * dt

        deformation = lsetmeshadap.CalcDeformation(exact.phi)

        # solve elliptic problem on a fixed surface to get u with normal extension
        InterpolateToP1(exact.phi, lset_approx)
        ci = CutInfo(mesh, lset_approx)

        ba_IF.Clear()
        ba_IF |= ci.GetElementsOfType(IF)
        update_ba_IF_band(lset_approx, mesh, c_delta * dt * exact.maxvel, ba_IF_band)

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

        with TaskManager():
            l2u, h1u = errors_vec(mesh, ds, Pmat, gfu_prevs[j], exact.u)
        mass_append(keys=keys, vals=[t_curr, l2u, h1u, 0, 0], **out_errs)

    dofs = []
    # TIME MARCHING
    exact.set_time(0.0)
    t_curr = 0.0

    if out:
        gfu_out.Set(gfu_prevs[0])
        gfp_out.Set(exact.p)
        vtk.Do(time=t_curr)

    i = 1

    while t_curr < tfinal - dt:
        exact.set_time(t_curr + dt)
        t_curr += dt
        with TaskManager():
            deformation = lsetmeshadap.CalcDeformation(exact.phi)

            InterpolateToP1(exact.phi, lset_approx)
            ci = CutInfo(mesh, lset_approx)

            ba_IF.Clear()
            ba_IF |= ci.GetElementsOfType(IF)
            update_ba_IF_band(lset_approx, mesh, c_delta * dt * exact.maxvel, ba_IF_band)

            VG = Compress(V, GetDofsOfElements(V, ba_IF_band))
            QG = Compress(Q, GetDofsOfElements(Q, ba_IF))
            dofs.append(VG.ndof + QG.ndof)

            u, v = VG.TnT()
            p, q = QG.TnT()

            # helper grid functions
            n_k, Hmat = helper_grid_functions(mesh=mesh, order=order, levelset=exact.phi, vel_space=V)

        gfu_approx = GridFunction(VG)
        if time_order == 1:
            gfu_approx.Set(Pmat * gfu_prevs[0])
        elif time_order == 2:
            gfu_approx.Set(2 * Pmat * gfu_prevs[0] - Pmat * gfu_prevs[1])
        else:
            gfu_approx.Set(3 * Pmat * gfu_prevs[0] - 3 * Pmat * gfu_prevs[1] + Pmat * gfu_prevs[2])

        a = BilinearForm(VG, symmetric=False)
        a += bdf_coeff[0]/dt * InnerProduct(u, Pmat * v) * ds
        a += exact.wN * InnerProduct(Hmat * u, Pmat * v) * ds
        a += 0.5 * InnerProduct((Pmat * grad(u) * Pmat - (u * n) * Hmat) * gfu_approx, v) * ds
        a += (-0.5) * InnerProduct((Pmat * grad(v) * Pmat - (v * n) * Hmat) * gfu_approx, u) * ds
        a += (-0.5) * InnerProduct(exact.g * u, Pmat * v) * ds
        a += 2.0 * exact.mu * (InnerProduct(Pmat * Sym(grad(u)) * Pmat - (u * n) * Hmat,
                                      Pmat * Sym(grad(v)) * Pmat - (v * n) * Hmat)) * ds
        a += tau * InnerProduct(n_k, u) * InnerProduct(n_k, v) * ds
        a += rho_u * InnerProduct(grad(u) * n, grad(v) * n) * dX

        # pressure mass-convection-total_stab_tests_diffusion matrix
        ap = BilinearForm(QG, symmetric=False)
        # mass part
        ap += bdf_coeff[0]/dt * p * q * ds
        # total_stab_tests_diffusion
        ap += 2 * exact.mu * InnerProduct(Pmat * grad(p), Pmat * grad(q)) * ds
        # convection
        ap += InnerProduct(Pmat * grad(p), Pmat * gfu_approx) * q * ds
        # normal gradient in the bulk stabilization
        # SHOULD IT BE rho_p OR rho_u?
        ap += rho_p * ((grad(p) * n) * (grad(q) * n)) * dX2

        # pressure total_stab_tests_diffusion matrix
        pd = BilinearForm(QG, symmetric=True)
        # total_stab_tests_diffusion
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
            f += (-1.0) * bdf_coeff[j+1]/dt * InnerProduct(gfu_prevs[j], Pmat * v) * ds

        g = LinearForm(QG)
        g += (-1.0) * exact.g * q * ds

        with TaskManager():
            pre_a = Preconditioner(a, "bddc")
            pre_pd = Preconditioner(pd, "bddc")
            pre_sq = Preconditioner(sq, "bddc")

            a.Assemble()
            ap.Assemble()
            pd.Assemble()
            b.Assemble()
            c.Assemble()
            pd.Assemble()
            sq.Assemble()

            f.Assemble()
            g.Assemble()

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
