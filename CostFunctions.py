from Helpers import normalizeSparse
from Ramp import rungeKuttaRampNew, rungeKuttaStep
from SimulationMain import initialiseSimulation
from sympy import *
import numpy as np


def maxFidelityCostFunction(grad, simulationparameters=initialiseSimulation(13, 'AFM', 'forward')
                            , dt=0.01, p=0.01):

    initialState, H, Htar, V, F = simulationparameters

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
        # Renormalizing the state
        currentState = normalizeSparse(currentState)

        # Transforming the state into the space to calculate fidelity
        currentState_f = F.transpose() * V * currentState

        # Appending current fidelity to array
        f.append(abs(currentState_f).power(2).sum())

        t_curr += dt

    return -np.max(f)
