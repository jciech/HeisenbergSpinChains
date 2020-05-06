import numpy as np
from sympy import *
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.abc import x

f = implemented_function('f', lambda x: np.random.normal(0, sqrt(x)))
lam_f = lambdify(x, f(x), modules=['numpy'])

simTime, gradient = symbols('simTime gradient')


def tanhRamp(t, g, tmax):
    """
    Computes the value of the ramp at a time t for a fixed gradient g and domain [0;tmax]

    :param t: specific value of t at which we calculate the value of the function
    :param g: gradient of the ramp
    :param tmax: specifies the implied domain as in the docstring
    :return:
    """
    return 0.5 * (tanh((t - tmax / 2) / g) + 1)


def tanhRampNoisy(t, g, n):
    return (1 + (n * lam_f(t))) * 0.5 * (tanh((t + (2 * g * atanh((2 * 0.01) - 1) / 2)) / g) + 1)


def rungeKuttaRamp(t, dt, grad, tmax):
    """
    The function gives a tuplet of parameters to feed into the rungeKuttaStep function for the evolution of the
    Hamiltonian at 3  different timesteps for a given gradient parameter and tmax.

    :param t: time at which the tuple is generated
    :param dt: timestep
    :param grad: gradient parameter in the ramp
    :param tmax: max time of the ramp
    :return: parameters for H at t, t+dt/2 and t+dt
    """

    return [float(tanhRamp(simTime, grad, tmax).subs(simTime, t)),
            float(tanhRamp(simTime, grad, tmax).subs(simTime, t + dt / 2)),
            float(tanhRamp(simTime, grad, tmax).subs(simTime, t + dt))]


def tanhRampNew(t, g, p):
    """
    Computes the value of the ramp at a time t for a fixed gradient g and, starting value p and final value 1-p

    :param t: specific value of t at which we calculate the value of the function
    :param g: gradient of the ramp
    :param p: specifies the initial and final value of tanhRampNew, also specifying its range
    :return:
    """
    return 0.5 * (tanh((t + (2 * g * atanh((2 * p) - 1) / 2)) / g) + 1)


def rungeKuttaRampNew(t, dt, grad, p):
    """
    The function gives a tuplet of parameters to feed into the rungeKuttaStep function for the evolution of the
    Hamiltonian at 3  different timesteps for a given gradient parameter.

    :param t: time at which the tuple is generated
    :param dt: timestep
    :param grad: gradient parameter in the ramp
    :return: parameters for H at t, t+dt/2 and t+dt
    """

    return [float(tanhRampNew(simTime, grad, p).subs(simTime, t)),
            float(tanhRampNew(simTime, grad, p).subs(simTime, t + dt / 2)),
            float(tanhRampNew(simTime, grad, p).subs(simTime, t + dt))]


def generateNoisyRamp2(g, dt=0.005, p=0.01, n=0.03):
    tmax = -2 * atanh(2 * p - 1)
    ramp = []
    t_curr = 0
    while t_curr < 5:
        p = float(tanhRampNoisy(t_curr, g, n))
        if p > 0:
            if p < 1:
                ramp.append(p)
            else:
                ramp.append(1)
        else:
            ramp.append(0)
        t_curr += dt
    return ramp


def rungeKuttaStep(state, Hamiltonian, Hamiltonian_later, Hamiltonian_latest, dt):
    """
    The function evolves the input state by the input Hamiltonian for a timestep dt using the Runge-Kutta
    approximation

    :param state: input state as vector
    :param Hamiltonian: input Hamiltonian
    :param Hamiltonian_later: Hamiltinian evolved by dt/2
    :param Hamiltonian_later: Hamiltinian evolved by dt
    :param dt: timestep
    :return: evolved state after a timestep dt
    """

    k1 = -1j * dt * Hamiltonian * state
    k2 = -1j * dt * Hamiltonian_later * (state + k1 / 2)
    k3 = -1j * dt * Hamiltonian_later * (state + k2 / 2)
    k4 = -1j * dt * Hamiltonian_latest * (state + k3)

    return state + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
