from ngsolve import SetNumThreads
import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import json

from utils import *
from laplace_solvers import moving_diffusion
from exact import Exact

SetNumThreads(16)

f = open('input/input_evolving_surface_diffusion_test.json')
args = json.load(f)

mode = args['mode']
order = args['order']
time_order = args['time_order']
alpha = args['alpha']
nu = args['nu']
tfinal = args['tfinal']
stab_type = args['stab_type']
unif_ref = args['unif_ref']
max_nref = args['max_nref']
fname = args['fname']
plt_out = args['plt_out']

if mode == 'l4l2':
    R = 1.0
    maxvel = 0.5
    params = {"nu": nu, "R": R, "maxvel": maxvel}
    exact = Exact(params)
    t = exact.t
    cfs = {
        "phi": CoefficientFunction(-R + (2*t*(x**2*y**2 + x**2*z**2 + y**2*z**2) + x**4 + y**4 + z**4)**(1/4)),
        "w": CoefficientFunction((
            -x*(t*(y**2 + z**2) + x**2)*(x**2*(y**2 + z**2) + y**2*z**2)/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)),
            -y*(t*(x**2 + z**2) + y**2)*(x**2*(y**2 + z**2) + y**2*z**2)/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)),
            -z*(t*(x**2 + y**2) + z**2)*(x**2*(y**2 + z**2) + y**2*z**2)/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6))
        )),
        "u": CoefficientFunction(
            (sin(pi * t) + 1) * sin(pi * x)
        ),
        "f": CoefficientFunction(
            (pi*x*(2*nu*(t**3*x**6*y**2 + t**3*x**2*y**6 + 3*t**2*x**6*y**2 + 4*t**2*x**4*y**4 + 3*t**2*x**2*y**6 + 2*t*x**8 + t*x**6*y**2 + 8*t*x**4*y**4 + t*x**2*y**6 + 2*t*y**8 + 2*t*z**8 + 2*t*z**4*(x**4*(2*t + 4) + x**2*y**2*(-t*(t - 7) + 6) + y**4*(2*t + 4)) + 3*x**6*y**2 + 3*x**2*y**6 + z**6*(t + 3)*(t**2 + 1)*(x**2 + y**2) + z**2*(x**2 + y**2)*(x**4*(t + 3)*(t**2 + 1) - x**2*y**2*(t + 1)*(t*(3*t - 14) + 3) + y**4*(t + 3)*(t**2 + 1))) - (x**2*y**2 + z**2*(x**2 + y**2))*(t*x**4*y**2*(t + 2) + t*x**2*y**4*(t + 2) + t*z**4*(t + 2)*(x**2 + y**2) + t*z**2*(6*t*x**2*y**2 + x**4*(t + 2) + y**4*(t + 2)) + x**6 + y**6 + z**6))*(t*(y**2 + z**2) + x**2)*(sin(pi*t) + 1)*cos(pi*x) + ((sin(pi*t) + 1)*(2*pi**2*nu*t**4*x**8*y**4 + 2*pi**2*nu*t**4*x**6*y**6 + 4*pi**2*nu*t**3*x**8*y**4 + 8*pi**2*nu*t**3*x**6*y**6 + 4*pi**2*nu*t**3*x**4*y**8 + 2*pi**2*nu*t**2*x**10*y**2 + 8*pi**2*nu*t**2*x**6*y**6 + 12*pi**2*nu*t**2*x**4*y**8 + 2*pi**2*nu*t**2*x**2*y**10 + 4*pi**2*nu*t*x**8*y**4 + 4*pi**2*nu*t*x**4*y**8 + 8*pi**2*nu*t*x**2*y**10 + 2*pi**2*nu*x**6*y**6 + 2*pi**2*nu*y**12 + 2*pi**2*nu*z**12 - t**3*x**8*y**4 - t**3*x**4*y**8 - 3*t**2*x**8*y**4 - 4*t**2*x**6*y**6 - 3*t**2*x**4*y**8 - 2*t*x**10*y**2 - t*x**8*y**4 - 8*t*x**6*y**6 - t*x**4*y**8 - 2*t*x**2*y**10 + 2*t*z**10*(x**2*(pi**2*nu*(t + 4) - 1) + y**2*(2*pi**2*nu*(t + 2) - 1)) - 3*x**8*y**4 - 3*x**4*y**8 + z**8*(x**4*(t*(4*pi**2*nu - 1)*(t*(t + 3) + 1) - 3) + x**2*y**2*(2*t*(t*(pi**2*nu*(t*(t + 6) + 18) - t - 3) - 2) - 6) + y**4*(t*(2*pi**2*nu*(t + 2)*(t*(t + 2) + 2) - t*(t + 3) - 1) - 3)) + z**6*(x**6*(2*pi**2*nu*(t**2*(t + 2)**2 + 1) - 4*t*(t + 2)) + x**4*y**2*(t*(4*pi**2*nu*(t*(t*(3*t + 13) + 3) + 1) + t*(t - 21) - 21) - 3) + x**2*y**4*(t*(2*pi**2*nu*(t*(t*(11*t + 26) + 9) + 4) + t*(t - 21) - 21) - 3) + y**6*(4*pi**2*nu*(t**2*(t + 2)**2 + 1) - 4*t*(t + 2))) + z**4*(2*t*x**4*y**4*(4*pi**2*nu*t*(t*(7*t + 4) + 4) + t*(3*t - 21) - 18) + x**8*(t*(2*pi**2*nu*(t**2*(t + 2) + 2) - t*(t + 3) - 1) - 3) + x**6*y**2*(t*(2*pi**2*nu*(t*(t*(11*t + 12) + 5) + 2) + t*(t - 21) - 21) - 3) + x**2*y**6*(t*(2*pi**2*nu*(t*(t*(11*t + 26) + 9) + 4) + t*(t - 21) - 21) - 3) + y**8*(t*(2*pi**2*nu*(t + 2)*(t*(t + 2) + 2) - t*(t + 3) - 1) - 3)) + z**2*(2*t*x**10*(pi**2*nu*t - 1) + 2*t*y**10*(2*pi**2*nu*(t + 2) - 1) + x**8*y**2*(2*t*(t*(2*pi**2*nu*(t*(t + 2) + 2) - t - 3) - 2) - 6) + x**6*y**4*(t*(2*pi**2*nu*(t*(t*(11*t + 12) + 5) + 2) + t*(t - 21) - 21) - 3) + x**4*y**6*(t*(4*pi**2*nu*(t*(t*(3*t + 13) + 3) + 1) + t*(t - 21) - 21) - 3) + x**2*y**8*(2*t*(t*(pi**2*nu*(t*(t + 6) + 18) - t - 3) - 2) - 6))) + 2*pi*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**2*cos(pi*t))*sin(pi*x))/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**2)
        ),
        "divGw": CoefficientFunction(
            -(x**2*(y**2 + z**2) + y**2*z**2)*(t**3*x**6*y**2 + t**3*x**2*y**6 + 3*t**2*x**6*y**2 + 4*t**2*x**4*y**4 + 3*t**2*x**2*y**6 + 2*t*x**8 + t*x**6*y**2 + 8*t*x**4*y**4 + t*x**2*y**6 + 2*t*y**8 + 2*t*z**8 + 2*t*z**4*(x**4*(2*t + 4) + x**2*y**2*(-t*(t - 7) + 6) + y**4*(2*t + 4)) + 3*x**6*y**2 + 3*x**2*y**6 + z**6*(t + 3)*(t**2 + 1)*(x**2 + y**2) + z**2*(x**2 + y**2)*(x**4*(t + 3)*(t**2 + 1) - x**2*y**2*(t + 1)*(t*(3*t - 14) + 3) + y**4*(t + 3)*(t**2 + 1)))/(2*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**2)
        ),
        "divGwT": CoefficientFunction(0.0),
        "fel": CoefficientFunction(
            (pi*x*(t*(y**2 + z**2) + x**2)*(t**3*x**6*y**2 + t**3*x**2*y**6 + 3*t**2*x**6*y**2 + 4*t**2*x**4*y**4 + 3*t**2*x**2*y**6 + 2*t*x**8 + t*x**6*y**2 + 8*t*x**4*y**4 + t*x**2*y**6 + 2*t*y**8 + 2*t*z**8 + 2*t*z**4*(x**4*(2*t + 4) + x**2*y**2*(-t*(t - 7) + 6) + y**4*(2*t + 4)) + 3*x**6*y**2 + 3*x**2*y**6 + z**6*(t + 3)*(t**2 + 1)*(x**2 + y**2) + z**2*(x**2 + y**2)*(x**4*(t + 3)*(t**2 + 1) - x**2*y**2*(t + 1)*(t*(3*t - 14) + 3) + y**4*(t + 3)*(t**2 + 1)))*cos(pi*x) + (t*x**4*(y**2 + z**2)*(t + pi**2*t + 2) + t*x**2*(t*y**2*z**2*(6 + 4*pi**2) + y**4*(t + 2 + 2*pi**2) + z**4*(t + 2 + 2*pi**2)) + x**6 + (1 + pi**2)*(y**2 + z**2)*(y**4 + y**2*z**2*(t*(t + 2) - 1) + z**4))*(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)*sin(pi*x))*(sin(pi*t) + 1)/(t*x**4*(t + 2)*(y**2 + z**2) + t*x**2*(6*t*y**2*z**2 + y**4*(t + 2) + z**4*(t + 2)) + t*y**4*z**2*(t + 2) + t*y**2*z**4*(t + 2) + x**6 + y**6 + z**6)**2
        )
    }
    exact.set_cfs(cfs)
    bbox_sz = 2.0
    band_type = 'inner'
elif mode == 'translation':
    R = 1.0
    nu = R * R
    maxvel = 0.2
    params = {"nu": nu, "R": R, "maxvel": maxvel}
    w1, w2, w3 = maxvel, 0, 0
    exact = Exact(params)
    t = exact.t
    cfs = {
        "phi": CoefficientFunction(sqrt((x - w1 * t) * (x - w1 * t) + y * y + z * z) - R),
        "w": CoefficientFunction((w1, w2, w3)),
        "u": CoefficientFunction(1 + (x + y + z - w1 * t) * exp(-2 * t)),
        "f": CoefficientFunction(0.0),
        "divGw": CoefficientFunction(0.0),
        "divGwT": CoefficientFunction(-2 * maxvel / (R * R) * (x - maxvel * t)),
        "fel": CoefficientFunction((-t**3*w1*(w1**2 + w2**2 + w3**2) + t**2*(w1**2*(3*x + y + z) + 2*w1*(w2*y + w3*z) + (w2**2 + w3**2)*(x + y + z) + (w1**2 + w2**2 + w3**2)*exp(2*t)) - 2*t*(w1*x + w2*y + w3*z)*exp(2*t) - t*(w1*(3*x**2 + 2*x*(y + z) + y**2 + z**2 + 2) + 2*w2 + 2*w3 + (w2*y + w3*z)*(2*x + 2*y + 2*z)) + 2*x + 2*y + 2*z + (x**2 + y**2 + z**2)*(x + y + z + exp(2*t)))*exp(-2*t)/(t**2*(w1**2 + w2**2 + w3**2) - 2*t*(w1*x + w2*y + w3*z) + x**2 + y**2 + z**2))
    }
    exact.set_cfs(cfs)
    bbox_sz = 2.0
    band_type = 'both'
else:
    print("Mode not recognized.")
    exit(1)

sns.set()

l2us = []
h1us = []

for nref in range(max_nref):
    exact.set_time(0.0)
    mesh = background_mesh(unif_ref=unif_ref, bbox_sz=bbox_sz)
    h = bbox_sz * 2 ** (1 - unif_ref - nref)
    dt = h ** ((order + 1) / time_order) / 4

    maxvel = exact.params['maxvel']
    phi = exact.cfs['phi']

    c_delta = time_order + 0.1
    band = maxvel * (tfinal + c_delta*dt)
    miniband = c_delta * dt * maxvel
    refine_around_lset(mesh, nref, phi, band, miniband, band_type=band_type)

    ndof, ts, l2uss, h1uss = moving_diffusion(mesh=mesh, dt=dt, order=order, tfinal=tfinal,
                                              exact=exact, band=miniband, time_order=time_order, stab_type=stab_type)

    if plt_out:
        plt.plot(l2uss)
        plt.title('L^2 vels')
        plt.show()

        plt.plot(h1uss)
        plt.title('H^1 vels')
        plt.show()

    l2u = max(l2uss)
    h1u = np.sqrt(sci.simps(y=np.array(h1uss)**2, x=ts, dx=dt, even='avg'))

    print(f"h = {h}, avg ndof = {int(ndof)}")

    if len(l2us) > 0:
        print(f"{ndof:.2E} & {np.log2(l2us[-1]/l2u):.2f} & {l2u:.2E} & {np.log2(h1us[-1]/h1u):.2f} & {h1u:.2E}")
    else:
        print(f"  ndof   &      &   lil2u  &      &   l2h1u")
        print(f"{ndof:.2E} &      & {l2u:.2E} &      & {h1u:.2E}")

    l2us.append(l2u)
    h1us.append(h1u)
