from Helpers import normalizeSparse
from Ramp import rungeKuttaRampNew, rungeKuttaStep
from MatrixMethods import makeStateAFM, phaseCorrectAFM, makeState
from sympy import *
import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt


def initialiseSimulation(N, magneticorder, rampdir, k=6):
    if magneticorder == 'AFM':
        AFMState = makeStateAFM(N - 1, kopt=k)
        AFMState = phaseCorrectAFM(N - 1, AFMState)
        targetState = 0
        initialState_fs = scipy.sparse.kron([0, 1], AFMState).transpose()

        # Loading the sparse matrix V\dagger and
        V = scipy.sparse.load_npz('MatrixGeneration/V_' + str(N) + '_allJ_Sz_1subspace.npz')
        F = scipy.sparse.load_npz('MatrixGeneration/F_' + str(N) + '_allJ_Sz_1subspace.npz')

    elif magneticorder == 'FM':

        config = [0 for i in range(N - 1)]
        config.insert(0, 1)

        configtarget = [0 for i in range(N - 1)]
        configtarget.insert(len(configtarget), 1)

        initialState_fs = makeState(config).transpose()
        targetState_fs = makeState(configtarget).transpose()

        V = scipy.sparse.load_npz('MatrixGeneration/V_' + str(N) + '_allJ_Sz_' + str(-(N - 2)) + 'subspace.npz')
        F = scipy.sparse.load_npz('MatrixGeneration/F_' + str(N) + '_allJ_Sz_' + str(-(N - 2)) + 'subspace.npz')
        targetState = V.transpose() * targetState_fs

    # Transforming initial state into contracted space
    initialState = V.transpose() * initialState_fs

    H = scipy.sparse.load_npz('MatrixGeneration/Hinitial_' + str(N) + rampdir + magneticorder + '.npz')
    Htar = scipy.sparse.load_npz('MatrixGeneration/Htarget_' + str(N) + rampdir + magneticorder +'.npz')

    return initialState, targetState, H, Htar, V, F

def maxFidelityCostFunction(grad, magneticorder, simulationparameters=None, dt=0.01, p=0.01):
    initialState, targetState, H, Htar, V, F = simulationparameters

    # Calculating the simulation time
    T = -2 * grad * atanh(2 * p - 1)

    # Setting up the simulation
    currentState = initialState
    f = []
    t_curr = 0

    while t_curr < T:
        # Computing the proportions of the Hamiltonian at each timestep, along with the values needed to compute RK step

        ramp = rungeKuttaRampNew(t_curr, dt, grad, p)

        # Updating the Hamiltonian

        Hcurr = (1 - ramp[0]) * H + ramp[0] * Htar
        H_dt2 = (1 - ramp[1]) * H + ramp[1] * Htar
        H_dt = (1 - ramp[2]) * H + ramp[2] * Htar

        # Performing the Runge-Kutta step
        currentState = rungeKuttaStep(currentState, Hcurr, H_dt2, H_dt, dt)

        # Renormalising the state
        currentState = normalizeSparse(currentState)
        if magneticorder == 'AFM':
            # Transforming the state into the space to calculate fidelity
            currentState_f = F.transpose() * V * currentState

            # Appending current fidelity to array
            f.append(abs(currentState_f).power(2).sum())

        elif magneticorder == 'FM':
            f.append(np.abs(np.dot(flatten(targetState.toarray()), flatten(currentState.toarray()))))

        t_curr += dt

    plt.plot([i * dt for i in range(len(f))], f)
    plt.show()
    return -np.max(f)