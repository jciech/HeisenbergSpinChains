import time
from CostFunctions import maxFidelityCostFunction, initialiseSimulation
from sympy import *
from scipy.optimize import shgo

if __name__ == "__main__":

    t_init = time.time()
    simTime, gradient = symbols("simTime gradient")
    order = "AFM"
    params = initialiseSimulation(11, order, "forward")
    print(
        maxFidelityCostFunction(
            3, "AFM", magneticorder=order, simulationparameters=params, dt=0.01, p=0.01
        )
    )
    elapsed = time.time() - t_init
    print(elapsed)

    # bounds = [(1, 20)]
    # result = scipy.optimize.shgo(maxFidelityCostFunction, bounds, (initialiseSimulation(13, 'AFM', 'forward'),
    # 0.01, 0.01), n=10, iters=1, disp=True,
    # sampling_method='sobol')
    # print(result)
