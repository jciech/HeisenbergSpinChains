import numpy as np
import scipy.sparse


def densityOperator(state):
    """
    The function take a state, in the form of an array, and creates a sparse matrix representing the density operator
    of that state

    :param state: array representing a state
    :return: Sparse matrix representing the density operator of a state
    """

    densOp = scipy.sparse.csr_matrix((0, state.shape[1]), dtype=complex)
    for i in range(state.shape[1]):
        densOp = scipy.sparse.vstack([densOp, state[:, i].A * state])

    return densOp.tocsr()


# Note, the below function does not work - use the quimb method instead if you want to use density matrices for
# any reason
def partialTrace(densOp, N, optimize=False):
    """
    Computes the reduced density matrix for a density operator by tracing out all qubits except the last.
    Inspired by https://scicomp.stackexchange.com/questions/30052/calculate-partial-trace-of-an-outer-product-in-python

    :param densOp: Matrix representing the density operator of an array of qubits
    :param N: Chain length
    :oaram optimize: np.einsum option, False by default
    :return: Matrix representing the density operator of an array of qubits keeping only the last qubit state
    """
    keep = np.asarray(N - 1)
    dims = [2 for i in range(N)]
    Ndim = len(dims)
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]
    rho_a = densOp.reshape(np.tile(dims, 2))
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)
