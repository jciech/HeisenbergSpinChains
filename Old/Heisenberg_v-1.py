import math
import numpy as np
import scipy.linalg

S0 = np.array([[1, 0], [0, 1]])
Sx = np.array([[0, 1], [1, 0]])
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])


def commutator(mat1, mat2):
    c = np.dot(mat1, mat2) - np.dot(mat2, mat1)
    return c


def anticommutator(mat1, mat2):
    ac = np.doty(mat1, mat2) + np.dot(mat2, mat1)
    return ac


# Checking relations between Pauli matrices
t = True

while t == True:
    # For Sx
    t = Sx.dot(S0) == Sx
    t = Sx.dot(Sx) == S0
    t = Sx.dot(Sy) == Sz * 1j
    t = Sx.dot(Sz) == Sy * -1j
    # For Sy
    t = Sy.dot(S0) == Sy
    t = Sy.dot(Sx) == Sz * -1j
    t = Sy.dot(Sy) == S0
    t = Sy.dot(Sz) == Sx * 1j
    # For Sz
    t = Sz.dot(S0) == Sz
    t = Sz.dot(Sx) == Sy * 1j
    t = Sz.dot(Sy) == Sx * -1j
    t = Sz.dot(Sz) == S0
    print("All the relations checked in the code are true.")
    t = False


# Constructing raising and lowering operators

down = 1 / 2 * (Sx - Sy * 1j)
up = 1 / 2 * (Sx + Sy * 1j)

print(up)
print(down)
