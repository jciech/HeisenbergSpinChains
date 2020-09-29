import math
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

# Defining the Pauli matrices to be used in our calculation
S0 = np.array([[1, 0], [0, 1]])
Sx = np.array([[0, 1], [1, 0]])
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])

# Some variables chosen initally for potential calculation
# Note hbar will likely be taken to be 1. I am keeping it here should I ever need it.
dt = 1
hbar = 1.05 * np.power(10.0, -34)


def initialiseChain(N):
    # Initialising a down - up - up - up (...) state in Fock space
    state = [(i + 1) % 2 for i in range(2 ** N)]
    state[0] = 0
    state[1] = 1
    print(state)

    # We initialise a list which acts as labels of the i'th and i+1'th site in the chain
    # The necessity of this will become clear as the Hamiltonian is constructed
    labelList1 = [i for i in range(N)]
    labelList2 = [(i + 1) for i in range(N)]

    return state, labelList1, labelList2


def makeHamiltonian(N, labelList1, labelList2):

    # We initialise the Hamiltonian as a 2^Nx2^N matrix with complex values for generality
    # Note that this is initialised as a sparse array to save memory
    H = scipy.sparse.csr_matrix((2 ** N, 2 ** N), dtype=complex)

    # We must loop over all nearest neighbour interactions to develop the Hamiltonian
    for interaction in range(N - 1):
        # Choosing the labels of the spins participating in the interaction
        spin1 = labelList1[interaction]
        spin2 = labelList2[interaction]

        # Initialising the products which will be used to calculate the Hamiltonian matrix elements
        ProdX = 1
        ProdY = 1
        ProdZ = 1

        # The computation of the matrix elements is as follows:
        # (Almost) every interaction gains a contribution from a pair of spins, contributing an Sx, Sy and Sz term to H
        # There are N-1 interactions and for each one, we add a term which is a Kronecker product of Pauli matrices
        # If a spin participates in this interaction.
        # It's clear we are looking at nearest neighbours below
        for site in range(N):
            if site == spin1 or site == spin2:
                ProdX = scipy.sparse.kron(ProdX, Sx, format="csr")
                ProdY = scipy.sparse.kron(ProdY, Sy, format="csr")
                ProdZ = scipy.sparse.kron(ProdZ, Sz, format="csr")
            else:
                ProdX = scipy.sparse.kron(ProdX, S0, format="csr")
                ProdY = scipy.sparse.kron(ProdY, S0, format="csr")
                ProdZ = scipy.sparse.kron(ProdZ, S0, format="csr")

        H += ProdX + ProdY + ProdZ

    return H


def evolveState(state, hamiltonian, timestep):

    evmat = scipy.linalg.expm(hamiltonian * -1j * timestep)

    newstate = evmat.dot(state)

    return newstate


if __name__ == "__main__":
    # We initialise the chain for a set number of spins
    N = 3
    param = initialiseChain(N)
    # We create the Hamiltonian
    H = makeHamiltonian(N, param[1], param[2])
    # And evolve the state
    finalState = evolveState(param[0], H, 100 * dt)
    counter = 0
    finalStateList = []
    # while (counter < 1000):
    #   finalState = evolveState(finalState, H, dt)
    #  counter += 1
    # finalStateList.append(finalState[2**N - 1])

    # After a certain time, the chain is supposed to evolve.
    # I see however now, that there is a problem with the encoding of states both in Fock space and as a complex number
    # I need to look for a way to represent this operation better.
    print(finalState)
    print(H)

    # When I have a range of N in line 36; the line below shows lengths which are unequal to 1
    # Moreover, the evolution never stops. From the below I remembered that there are N-1 interactions.
    # I'm leaving this here as a testament to simple mistakes :)
    print(np.sqrt(np.real(finalState[0]) ** 2 + np.imag(finalState[0]) ** 2))
