from CostFunctions import maxFidelityCostFunction, initialiseSimulation
from MatrixMethods import *
from scipy.optimize import shgo

if __name__ == '__main__':

    simTime, gradient = symbols('simTime gradient')
    params = initialiseSimulation(13, 'AFM', 'forward')
    print(maxFidelityCostFunction(12, simulationparameters=params, dt=0.01, p=0.01))

    #bounds = [(1, 100)]
    #result = scipy.optimize.shgo(maxFidelityCostFunction, bounds, (initialiseSimulation(13, 'AFM', 'forward'),
                                                                   #0.01, 0.01), n=10, iters=1, disp=True,
                                 #sampling_method='sobol')
    #print(result)










