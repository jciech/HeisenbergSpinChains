import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from CostFunctions import maxFidelityCostFunction
from MatrixMethods import *
from scipy.optimize import shgo


def initialiseSimulation(N, magneticorder, rampdir):
    if magneticorder == 'AFM':
        AFMState = makeStateAFM(N - 1)
        AFMState = phaseCorrectAFM(N - 1, AFMState)

        initialState_fs = scipy.sparse.kron([0, 1], AFMState).transpose()

        # Loading the sparse matrix V\dagger and
        V = scipy.sparse.load_npz('V_' + str(N) + '_allJ_Sz_1subspace.npz')
        F = scipy.sparse.load_npz('F_' + str(N) + '_allJ_Sz_1subspace.npz')

    elif magneticorder == 'FM':
        # TODO: Make this work for the ferromagnetic state (i.e. create necessary files and make sure the code runs)
        config = [0 for i in range(N)]
        initialState_fs = makeState().transpose()

        V = scipy.sparse.load_npz('V_' + str(N) + '_allJ_Sz_-1subspace.npz')
        F = scipy.sparse.load_npz('F_' + str(N) + '_allJ_Sz_-1subspace.npz')

    # Transforming initial state into contracted space
    initialState = V.transpose() * initialState_fs
    H = scipy.sparse.load_npz('Hinitial_' + str(N) + rampdir + '.npz')
    Htar = scipy.sparse.load_npz('Htarget_' + str(N) + rampdir + '.npz')

    return initialState, H, Htar, V, F

if __name__ == '__main__':

    simTime, gradient = symbols('simTime gradient')
    bounds = [(1, 100)]
    result = scipy.optimize.shgo(maxFidelityCostFunction, bounds, (initialiseSimulation(13, 'AFM', 'forward'),
                                                                   0.01, 0.01), n=10, iters=1, disp=True,
                                 sampling_method='sobol')
    print(result)










