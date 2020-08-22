# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import logging
import logging.config
import os
import shutil
import struct
import sys
import wave

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from adjoint_model_displacement import adjoint_model
from dae_solver import dae_solver
from fem_solver import vocal_tract_solver, vocal_tract_solver_backward
from ode_solver import ode_solver
from vocal_fold_model_displacement import vdp_coupled, vdp_jacobian

for path in ["models/vocal_fold", "solvers/ode_solvers", "solvers/pde_solvers"]:
    sys.path.append(
        os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), path)
    )


def pcm16_to_float(wav_file):
    """
    Convert 16-bit signed integer PCM wav samples to
    float array between [-1, 1].
    """
    w = wave.open(wav_file)

    num_frames = w.getnframes()
    num_channels = w.getnchannels()
    num_samples = num_frames * num_channels
    fs = w.getframerate()  # sample rate

    # Convert binary chunks to short ints
    fmt = "%ih" % num_samples
    raw_data = w.readframes(num_frames)
    a = struct.unpack(fmt, raw_data)

    # Convert short ints to floats
    a = [float(val) / pow(2, 15) for val in a]

    return np.array(a, dtype=float), fs


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-cf", "--configure_file", required=True, help="configure file for experiment")
args = parser.parse_args()

# Load configures
configure_file = args.configure_file
try:
    with open(configure_file) as f:
        configs = json.load(f)
except OSError as e:
    print(f"OS error: {e}")

# Log
log_dir = os.path.join(configs["project_root"], configs["log_dir"])
try:
    os.makedirs(log_dir)
except FileExistsError:
    print(f"folder {log_dir} already exists")
log_file = configs["log"]["handlers"]["file"]["filename"]
if os.path.isfile(log_file):
    print(f"file {log_file} already exists")
    raise FileExistsError
# Setup logger
logging.config.dictConfig(configs["log"])
logger = logging.getLogger("main")
# Copy configure file to log dir
try:
    target_file = os.path.join(
        log_dir,
        os.path.basename(log_file) + ".configure.json" + f".{datetime.datetime.now().date()}",
    )
    shutil.copyfile(configure_file, target_file)
except IOError as e:
    logger.exception(f"Unable to copy file: {e}")
logger.info(f"Copied {configure_file} to {target_file}")
if len(sys.argv) < 2:
    print(f"python {sys.argv[0]} configure.json")
    sys.exit(-1)

np.random.seed(configs["random_seed"])
logger.info(f"Set random seed to: {configs['random_seed']}")

# Data
wav_file = configs["wav_file"]
samples, fs = pcm16_to_float(wav_file)
assert int(fs) == 8000, f"{wav_file}: incompatible sampling rate, need 8000 but got {fs}"

fig = plt.figure()
plt.plot(np.linspace(0, len(samples) / fs, len(samples)), samples)
plt.show()

# Trim: optional
samples = samples[int(4 * fs) : int(4 * fs) + 100]

samples = samples / np.linalg.norm(samples)  # normalize

# Initial conditions
alpha = 0.8  # if > 0.5 delta, stable-like oscillator
beta = 0.32
delta = 1.0  # asymmetry parameter

vdp_init_t = 0.0
vdp_init_state = [0.0, 0.1, 0.0, 0.1]  # (xr, dxr, xl, dxl), xl=xr=0

logger.info(f"Initial parameters: alpha = {alpha:.4f}   beta = {beta:.4f}   delta = {delta:.4f}")
logger.info("-" * 110)

# Model related constants
x0 = 0.1  # half glottal width at rest position, cm
d = 1.75  # length of vocal folds, cm
tau = 1e-3  # time delay for surface wave to travel half glottal height T, 1 ms
eta = 1.0  # nonlinear factor for energy dissipation at large amplitude
c = 5000  # air particle velocity, cm/s
M = 0.5  # mass, g/cm^2
B = 100  # damping, dyne s/cm^3

c_sound = 34000.0  # speed of sound, cm/s
tau_f = 1.0  # parameter for updating f
gamma_f = 1.0  # parameter for updating f

# FEM related constants
Nx = 64  # number of uniformly spaced cells in mesh
BASIS_DEGREE = 2  # degree of the basis functional space
length = 17.5  # spatial dimension, length of vocal tract, cm
divisions = (Nx,)  # mesh size
num_dof = Nx * BASIS_DEGREE + 1  # degree of freedom

num_tsteps = len(samples)  # number of time steps
T = len(samples) / float(fs)  # total time, s
dt = T / num_tsteps  # time step size
logger.info(f"Total time: {T:.4f}s  Stepsize: {dt:.4g}s")

f_data = np.zeros((num_tsteps, num_dof))  # init f
R = 1e16  # prediction residual w.r.t L2 norm
iteration = 0

while R > 0.1:

    # Step 1: solve vocal fold displacement model
    logger.info("Solving vocal fold displacement model")

    vdp_params = [alpha, beta, delta]

    K = B**2 / (beta**2 * M)
    Ps = (alpha * x0 * np.sqrt(M * K)) / tau

    time_scaling = np.sqrt(K / float(M))  # t -> s
    x_scaling = np.sqrt(eta)

    logger.debug(
        f"stiffness K = {K:.4f} dyne/cm^3    subglottal Ps = {Ps:.4f} dyne/cm^2    time_scaling = {time_scaling:.4f}"
    )

    # Solve vocal fold displacement model
    sol = ode_solver(
        vdp_coupled,
        vdp_jacobian,
        vdp_params,
        vdp_init_state,
        vdp_init_t,
        solver="lsoda",
        ixpr=0,
        dt=(time_scaling / float(fs)),  # dt -> ds
        tmax=(time_scaling * len(samples) / float(fs)),
    )

    # Plot xl, xr for 0.1s
    plt.figure()
    plt.plot(sol[: int(0.1 * fs), 0], sol[: int(0.1 * fs), 1], "b.-")
    plt.plot(sol[: int(0.1 * fs), 0], sol[: int(0.1 * fs), 3], "r.-")
    plt.legend(["right", "left"])
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()

    # Plot states
    plt.figure()
    plt.subplot(121)
    plt.plot(sol[:, 1], sol[:, 3], "b.-")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.subplot(122)
    plt.plot(sol[:, 2], sol[:, 4], "b.-")
    plt.xlabel("du")
    plt.ylabel("dv")
    plt.show()

    # Step 2: solve forward vocal tract model
    logger.info("Solving forward vocal tract model")

    # Calculate some terms
    if len(sol) > len(samples):
        sol = sol[:-1]
    assert len(sol) == len(
        samples
    ), f"Inconsistent length: ODE sol ({len(sol):d}) / wav file ({len(samples):d})"

    X = sol[:, [1, 3]]  # vocal fold displacement (right, left), cm
    dX = sol[:, [2, 4]]  # cm/s
    u0 = c * d * (np.sum(X, axis=1) + 2 * x0)  # volume velocity flow, cm^3/s
    u0 = u0 / np.linalg.norm(u0)  # normalize
    u0 = u0[:num_tsteps]

    plt.figure()
    plt.plot(np.linspace(0, T, len(u0)), u0)
    plt.show()

    uL_k, U_k = vocal_tract_solver(
        f_data, u0, samples, c_sound, length, Nx, BASIS_DEGREE, T, num_tsteps, iteration
    )

    # Step 3: calculate difference signal
    logger.info("Calculating difference signal")
    uL_k = uL_k / np.linalg.norm(uL_k)  # normalize
    r_k = samples - uL_k

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, T, len(samples)), samples, "b")
    ax.plot(np.linspace(0, T, len(samples)), uL_k, "r")
    ax.set_xlabel("t")
    plt.legend(["samples", "uL_k"])
    plt.tight_layout()
    plt.show()

    # Step 4: solve backward vocal tract model
    logger.info("Solving backward vocal tract model")
    Z_k = vocal_tract_solver_backward(
        f_data, r_k, c_sound, length, Nx, BASIS_DEGREE, T, num_tsteps, iteration
    )

    # Step 5: update f
    logger.info("Updating f^k")
    f_data = f_data + (tau_f / gamma_f) * (Z_k[::-1, ...] / (c_sound**2) + U_k)

    iteration = iteration + 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    XX, YY = np.meshgrid(
        np.linspace(0, T, f_data.shape[0]), np.linspace(0, length, f_data.shape[1])
    )
    ax.plot_surface(XX, YY, f_data.T, cmap="coolwarm")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_zlabel("f")
    plt.tight_layout()
    plt.show()

    # Step 6: solve adjoint model
    logger.info("Solving adjoint model")

    dHf_u0 = uL_k / (u0 + 1e-14)  # derivative of operator Hf w.r.t. u0
    R_diff = 2 * c * d * (-r_k) * dHf_u0  # term c.r.t. difference signal

    residual, jac = adjoint_model(alpha, beta, delta, X, dX, R_diff, num_tsteps / T)

    M_T = [0.0, 0.0, 0.0, 0.0]  # initial states of the adjoint model at T
    dM_T = [0.0, -R_diff[-1], 0.0, -R_diff[-1]]  # initial ddL = ddE = -R_diff(T)

    adjoint_sol = dae_solver(
        residual,
        M_T,
        dM_T,
        T,
        solver="IDA",
        algvar=[0, 1, 0, 1],
        suppress_alg=True,
        atol=1e-6,
        rtol=1e-6,
        usejac=True,
        jac=jac,
        usesens=False,
        backward=True,
        tfinal=0.0,
        ncp=(len(samples) - 1),  # simulate (T --> 0)s backwards
        display_progress=True,
        report_continuously=False,
        verbosity=50,
    )  # NOTE: report_continuously should be False

    # Step 7: update vocal fold model parameters
    logger.info("Updating parameters")

    L = adjoint_sol[1][:, 0][::-1]  # reverse time 0 --> T
    E = adjoint_sol[1][:, 2][::-1]
    assert (len(L) == num_tsteps) and (len(E) == num_tsteps), "Size mismatch"
    L = L / np.linalg.norm(L)
    E = E / np.linalg.norm(E)

    d_alpha = -np.dot((dX[:num_tsteps, 0] + dX[:num_tsteps, 1]), (L + E))
    d_beta = np.sum(
        L * (1 + np.square(X[:num_tsteps, 0])) * dX[:num_tsteps, 0]
        + E * (1 + np.square(X[:num_tsteps, 1])) * dX[:num_tsteps, 1]
    )
    d_delta = np.sum(0.5 * (X[:num_tsteps, 1] * E - X[:num_tsteps, 0] * L))

    # Adaptive stepsize
    stepsize = 0.01 / np.max([d_alpha, d_beta, d_delta])
    alpha = alpha - stepsize * d_alpha
    beta = beta - stepsize * d_beta
    delta = delta - stepsize * d_delta

    R = np.sqrt(np.sum(r_k**2))

    logger.info(
        f"L2 Residual = {R:.4f} | alpha = {alpha:.4f}   beta = {beta:.4f}   delta = {delta:.4f}"
    )
    logger.info("-" * 110)

# Results
logger.info("*" * 110)

logger.info(
    f"alpha = {alpha:.4f}   beta = {beta:.4f}   delta = {delta:.4f}  L2 Residual = {R:.4f}"
)
