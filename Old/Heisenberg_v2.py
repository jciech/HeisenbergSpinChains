import copy
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

# Defining the Pauli matrices to be used in our calculation
S0 = np.array([[1, 0], [0, 1]])
Sx = np.array([[0, 1], [1, 0]])
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])

# Some variables chosen initally for potential calculation
# Note hbar will likely be taken to be 1. I am keeping it here should I ever need it.
dt = 1
hbar = 1.05 * np.power(10., -34)


def countBits(x):
    """
    The function takes counts the number of 1's in the binary representation of a base 10 integer.

    :param x: Base 10 integer
    :return: The number of ones of the integer in binary representation
    """
    # from https://stackoverflow.com/questions/10874012/how-does-this-bit-manipulation-work-in-java/10874449#10874449
    # Used because of the O(log(n)) complexity

    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x0000003F


def intToBinary(x, N):
    """
    The function converts a base 10 integer into a binary number of a specified length (padding with 0s if needed)

    :param x: Base 10 integer to be converted to binary
    :param N: Length of binary string required
    :return: Binary number of specified length
    """
    return ('{0:0' + str(N) + 'b}').format(x)


def makeState(configuration):
    """
    The function makeState takes argument configuration which specifies the state we want to initialise.

    :param configuration: np.array of length N with 1 meaning up and 0 meaning down, e.g. [1,0,0,1] means up-down-down-up
    :return: np.array of length 2**N, the state vector in full state space
    """
    # Initialising the state vector, which will be represented in the full state space of the chain
    state = 1

    # Defining the states from which our specific state will be constructed from
    up = [1, 0]
    down = [0, 1]

    # The loop below constructs a vector of size 2^N through the Kronecker product
    i = 0
    while i < np.length(configuration):
        if configuration[i] == 0:
            state = scipy.kron(state, down)
            i += 1
        elif configuration[i] == 1:
            state = scipy.kron(state, up)
            i += 1
        else:
            return "The configuration has not been specified correctly, please read the docstring"

    return state


def groupSz(N):
    """
    Creates a list of lists of states, in the full state space, grouped according to their total Sz spin

    :param N: Spin chain length as integer
    :return: List of lists of states with a given Sz for a chain of given length
    """
    # Indexing from -N to N translated into what can be handled by an array
    SzList = [[] for i in range(N + 1)]

    # There is room for optimisation here, the loop adds O(2^N) complexity
    for state in range(2 ** N):
        # The below is only a proxy for spin, it just creates an ordering for the purposes of the code
        totalSpin = countBits(state)
        SzList[totalSpin].append([int(bit) for bit in (intToBinary(state, N))])

    return SzList


def makeHamiltonian(N, initialState, SzList):
    """
    The function makeHamiltonian takes argument N, the length of our chain and constructs the XYZ Hamiltonian for
    a chain of that length, for a given initial state (to suppress unneeded data)


    :param N: integer length of the chain
    :param initialState: 2**N size vector specifying the initial state of the chain in the full space
    :return: 2**N x 2**N Hamiltonian of a chain of the specified length
    """
    H = scipy.zeros([2 ** N, 2 ** N], dtype=float)

    totalSpin = countBits(initialState)
    basis = SzList[totalSpin]
    for state in basis:
        print(state)
    # We must loop over all nearest neighbour interactions to develop the Hamiltonian
    for interaction in range(N - 1):
        # Initialising the products which will be used to calculate the Hamiltonian matrix elements
        ProdX = 1
        ProdY = 1
        ProdZ = 1

        # The computation of the matrix elements is as follows:
        # (Almost) every interaction gains a contribution from a pair of spins, contributing an Sx, Sy and Sz term to H
        # There are N-1 interactions and for each one, we add a term which is a Kronecker product of Pauli matrices
        # if a spin participates in this interaction. Otherwise we take the Kronecker product with the identity to
        # ensure correct dimensionality of the matrix
        # It's clear we are looking at nearest neighbours below
        for site in range(N):
            if site == interaction or site == interaction + 1:
                ProdX = scipy.kron(ProdX, Sx)
                ProdY = scipy.kron(ProdY, Sy)
                ProdZ = scipy.kron(ProdZ, Sz)
            else:
                ProdX = scipy.kron(ProdX, S0)
                ProdY = scipy.kron(ProdY, S0)
                ProdZ = scipy.kron(ProdZ, S0)

        H += np.real(ProdX + ProdY + ProdZ)

    return H

def findTargetState(initialConfiguration):

    targetConfiguration = initialConfiguration.append(initialConfiguration[0])
    targetConfiguration.remove(initialConfiguration[0])
    targetState = makeState(targetConfiguration)

    return targetState

def fidelity(targetConfiguration, state):
    """
    The function fidelity takes arguments initialState and state and calculates the fidelity between them.

    :param initialState: 2**N vector representing the initial state
    :param state: 2**N vector representing the state for which the calculation is conducted
    :return: float fidelity
    """
    targetState = findTargetState(targetConfiguration)
    f = np.dot(targetState, state)
    return np.abs(f)

def makeHamiltonianJ(N, J):
    """
    The function makeHamiltonian takes argument N, the length of our chain and constructs the XYZ Hamiltonian for
    a chain of that length. This particular function accomodates for non-uniform J coupling.

    :param N: integer length of the chain
    :param J: list of couplings of length N-1
    :return: 2**N x 2**N Hamiltonian of a chain of the specified length
    """
    H = scipy.zeros([2 ** N, 2 ** N], dtype=float)

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
                ProdX = scipy.kron(ProdX, Sx)
                ProdY = scipy.kron(ProdY, Sy)
                ProdZ = scipy.kron(ProdZ, Sz)
            else:
                ProdX = scipy.kron(ProdX, S0)
                ProdY = scipy.kron(ProdY, S0)
                ProdZ = scipy.kron(ProdZ, S0)

        H += np.real(Jtemp * (ProdX + ProdY + ProdZ))

    return H


def evolveState(state, hamiltonian, timestep):
    """
    The function evolveState takes an input state, as generated by initialiseChain, a Hamiltonian as generated by
    makeHamiltonian and timestep (integer or float) and returns a final state after constructing a time evolution
    operator.

    :param state: np.array of length 2**N, the state vector in full state space
    :param hamiltonian: 2**N x 2**N Hamiltonian of a chain of the specified length
    :param timestep: integer/float
    :return: evolved state: np.array of length 2**N, again in the full state space
    """
    evmat = scipy.linalg.expm(hamiltonian * -1j * timestep)
    newstate = evmat.dot(state)

    return newstate


############# TESTING BELOW ############################################################################################

config = [1,0,0]
initialState = makeState(config)
H = makeHamiltonian(3)

t = [i*dt for i in range(75)]
S = [[] for i in range(len(initialState))]
f = []
currentState = initialState
targetState = findTargetState(config)
for i in range(len(initialState)):
    for j in range(75):
        currentState = evolveState(initialState, H, 0.05 * j * dt)
        S[i].append(np.abs(currentState[i]))
        f.append(fidelity(targetState, currentState))

for state in S:
    plt.plot(t, state, label="S" + str(S.index(state)))

plt.plot(t, f[:75], label="fidelity")
plt.legend()
plt.show()
