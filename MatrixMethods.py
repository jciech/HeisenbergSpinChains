import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from Helpers import *
from sympy import *

# Defining the Pauli matrices to be used in our calculation
S0 = scipy.sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
Sx = scipy.sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
Sy = scipy.sparse.csr_matrix([[0.0, -1.0j], [1.0j, 0.0]])
Sz = scipy.sparse.csr_matrix([[1.0, 0.0], [0.0, -1.0]])


def makeHamiltonianJ(N, J):
    """
    The function makeHamiltonianJ takes argument N, the length of our chain and constructs the varying
    Hamiltonian for a chain of that length. This particular function accomodates for non-uniform J coupling.
    The variation in time is governed by tanhRamp. The couplings varied are J_1 (decreasing from 1) and J_N-1
    (increasing from 0)

    :param N: integer length of the chain
    :param J: list of couplings of length N-1
    :return: 2**N x 2**N sparse matrix Hamiltonian of a chain of the specified length
    """
    H = scipy.sparse.csr_matrix((2 ** N, 2 ** N))

    # We must loop over all nearest neighbour interactions to develop the Hamiltonian
    for interaction in range(N - 1):
        # Initialising the products which will be used to calculate the Hamiltonian matrix elements
        ProdX = 1
        ProdY = 1
        ProdZ = 1
        Jtemp = J[interaction]
        # The computation of the matrix elements is as follows:
        # (Almost) every interaction gains a contribution from a pair of spins, contributing an Sx, Sy and Sz term to H
        # There are N-1 interactions and for each one, we add a term which is a Kronecker product of Pauli matrices
        # if a spin participates in this interaction. Otherwise we take the Kronecker product with the identity to
        # ensure correct dimensionality of the matrix
        # It's clear we are looking at nearest neighbours below
        for site in range(N):
            if site == interaction or site == interaction + 1:
                ProdX = scipy.sparse.kron(ProdX, Sx, format="csr")
                ProdY = scipy.sparse.kron(ProdY, Sy, format="csr")
                ProdZ = scipy.sparse.kron(ProdZ, Sz, format="csr")
            else:
                ProdX = scipy.sparse.kron(ProdX, S0, format="csr")
                ProdY = scipy.sparse.kron(ProdY, S0, format="csr")
                ProdZ = scipy.sparse.kron(ProdZ, S0, format="csr")

        H += Jtemp * (ProdX + ProdY + ProdZ)

    return H


def makeHamiltonianPerturbed(N, J, h):
    """
    The function makeHamiltonianPerturbed takes argument N, the length of our chain and constructs the varying
    Hamiltonian for a chain of that length. This particular function accomodates for non-uniform J coupling.
    The variation in time is governed by tanhRamp. The couplings varied are J_1 (decreasing from 1) and J_N-1
    (increasing from 0). This version of the function computes the Hamiltonian under a small perturbation h of
    the magnetic field in the z direction. This is used to compute the ground state of the antiferromagnet.

    :param h: float magnitude of the perturbation
    :param N: integer length of the chain
    :param J: list of couplings of length N-1
    :return: 2**N x 2**N sparse matrix Hamiltonian of a chain of the specified length
    """
    H = scipy.sparse.csr_matrix((2 ** N, 2 ** N))

    # We must loop over all nearest neighbour interactions to develop the Hamiltonian
    for interaction in range(N - 1):
        # Initialising the products which will be used to calculate the Hamiltonian matrix elements
        ProdX = 1
        ProdY = 1
        ProdZ = 1
        Jtemp = J[interaction]
        hProd = 1
        # The computation of the matrix elements is as follows:
        # (Almost) every interaction gains a contribution from a pair of spins, contributing an Sx, Sy and Sz term to H
        # There are N-1 interactions and for each one, we add a term which is a Kronecker product of Pauli matrices
        # if a spin participates in this interaction. Otherwise we take the Kronecker product with the identity to
        # ensure correct dimensionality of the matrix
        # It's clear we are looking at nearest neighbours below

        # There is also a term which is responsible for the perturbation h at each site
        for site in range(N):
            if site == interaction:
                ProdX = scipy.sparse.kron(ProdX, Sx, format="csr")
                ProdY = scipy.sparse.kron(ProdY, Sy, format="csr")
                ProdZ = scipy.sparse.kron(ProdZ, Sz, format="csr")
                hProd = scipy.sparse.kron(hProd, Sz, format="csr")
            elif site == interaction + 1:
                ProdX = scipy.sparse.kron(ProdX, Sx, format="csr")
                ProdY = scipy.sparse.kron(ProdY, Sy, format="csr")
                ProdZ = scipy.sparse.kron(ProdZ, Sz, format="csr")
                hProd = scipy.sparse.kron(hProd, S0, format="csr")

            else:
                ProdX = scipy.sparse.kron(ProdX, S0, format="csr")
                ProdY = scipy.sparse.kron(ProdY, S0, format="csr")
                ProdZ = scipy.sparse.kron(ProdZ, S0, format="csr")
                hProd = scipy.sparse.kron(hProd, S0, format="csr")

        H += Jtemp * (ProdX + ProdY + ProdZ) - h * hProd

    return H


def makeState(configuration):
    """
    The function makeState takes argument configuration which specifies the state we want to initialise.

    :param configuration: np.array of length N with 1 meaning up and 0 meaning down, e.g. [1,0,0,1] means up-down-down-up
    :return: sparse matrix of length 2**N, the state vector in full state space
    """
    # Initialising the state vector, which will be represented in the full state space of the chain
    state = 1

    # Defining the states from which our specific state will be constructed from
    up = scipy.sparse.csr_matrix([0, 1])
    down = scipy.sparse.csr_matrix([1, 0])

    # The loop below constructs a vector of size 2^N through the Kronecker product
    i = 0
    while i < len(configuration):
        if configuration[i] == 0:
            state = scipy.sparse.kron(state, down, format="csr")
            i += 1
        elif configuration[i] == 1:
            state = scipy.sparse.kron(state, up, format="csr")
            i += 1
        else:
            return (
                "The configuration has not been specified correctly, please read the"
                " docstring"
            )

    return state


def makeStateArray(configuration):
    """
    The function makeStateArray takes argument configuration which specifies the state we want to initialise.

    :param configuration: np.array of length N with 1 meaning up and 0 meaning down, e.g. [1,0,0,1] means up-down-down-up
    :return: np.array of length 2**N, the state vector in full state space
    """
    # Initialising the state vector, which will be represented in the full state space of the chain
    state = 1

    # Defining the states from which our specific state will be constructed from
    up = [0, 1]
    down = [1, 0]

    # The loop below constructs a vector of size 2^N through the Kronecker product
    i = 0
    while i < len(configuration):
        if configuration[i] == 0:
            state = scipy.kron(state, down)
            i += 1
        elif configuration[i] == 1:
            state = scipy.kron(state, up)
            i += 1
        else:
            return (
                "The configuration has not been specified correctly, please read the"
                " docstring"
            )

    return state


def makeSubSpace(N, Sz):
    """
    The function creates a matrix of vectors in a particular region of the state space of the spin chain, with a
    particular value of Sz, i.e. a certain number of 1's and 0's in their binary representation. This will be used
    to contract the full Hamiltonian in order to make computations easier, as the Hamiltonian preserves Sz.

    :param N: Length of spin chain
    :param Sz: Total number of 1's in states
    :return: Sparse matrix defining the Sz subspace of the Hamiltonian
    """
    ss = scipy.sparse.csr_matrix((0, 2 ** N), dtype=int)
    for i in range(2 ** N - 1):
        if countBits(i) == Sz:
            ss = scipy.sparse.vstack(
                [ss, makeState([int(j) for j in intToBinary(i, N)])]
            )

    return ss.tocsr()


def makeAFMSubSpace(N, Sz):
    """
    The function creates a matrix of vectors in a particular region of the state space of the spin chain, with a
    particular value of Sz, i.e. a certain number of 1's and 0's in their binary representation and ending with a 1.
    This will be used to contract the Hamiltonian to the subspace in which we can compute the AFM state fidelity

    :param N: Length of spin chain
    :param Sz: Total number of 1's in states
    :return: Sparse matrix defining the Sz subspace of the Hamiltonian, with vectors ending in 1s only
    """
    ss = scipy.sparse.csr_matrix((0, 2 ** N), dtype=int)
    for i in range(2 ** N - 1):
        if countBits(i) == Sz:
            if intToBinary(i, N)[-1] == "1":
                ss = scipy.sparse.vstack(
                    [ss, makeState([int(j) for j in intToBinary(i, N)])]
                )

    return ss.tocsr()


def makeStateAFM(N, kopt=6):
    """
    The function computes the antiferromagnetic ground state by taking the lowest energy eigenvector for the
    Heisenberg Hamiltonian in the full state space for a chain of length N

    :param N: length of the chain
    :param kopt: optional - specify the number of eigenvalues to be solved for by eigsh (default 6, specify less for
    small chains
    :return: full state space representation of the antiferromagnetic ground state
    """
    H = makeHamiltonianJ(N, [1 for i in range(N - 1)])
    eigval, eigvec = scipy.sparse.linalg.eigsh(H, k=kopt, which="SA")
    return eigvec[:, 0]


def generateThermalStates(N, kopt=6):
    H = makeHamiltonianJ(N, [1 for i in range(N - 1)])
    eigval, eigvec = scipy.sparse.linalg.eigsh(H, k=kopt, which="SA")
    return [eigvec[:, i] for i in range(kopt)]


def phaseCorrectAFM(N, vec):
    """
    The function computes the global phase of a given antiferromagnetic ground state and reduces it to 0
    for a chain of length N

    :param N: length of the chain
    :param vec: full state space representation of antiferromagnetic ground state
    :return: Dephased representation of the antiferromagnetic ground state in the full space
    """
    V = scipy.sparse.load_npz("MatrixGeneration/V_" + str(N) + "_allJ_Sz_0subspace.npz")
    contVec = V.transpose() * vec
    phi = Symbol("phi")
    sol = solve(im(exp(-1j * phi) * contVec[0]), phi)[0]
    phase = complex((exp(-1j * sol[re(phi)])).evalf())

    return V * np.real(phase * contVec)
