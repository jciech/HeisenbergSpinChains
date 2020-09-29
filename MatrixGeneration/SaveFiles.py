import scipy.sparse
from MatrixMethods import makeHamiltonianJ, makeAFMSubSpace, makeSubSpace
import argparse


def saveMatrices(N, Sz):
    H_N = makeHamiltonianJ(N, [1 for i in range(N - 1)])
    V = makeSubSpace(N, Sz)
    F = makeAFMSubSpace(N, Sz)

    H_basis = V * H_N * V.transpose()

    scipy.sparse.save_npz(
        "V_" + str(N) + "_allJ_Sz_" + str(2 * Sz - N) + "subspace.npz", V.transpose()
    )
    # This saves V^\dagger to be able to transform basis vectors

    scipy.sparse.save_npz(
        "H_" + str(N) + "_allJ_Sz_" + str(2 * Sz - N) + "subspace.npz", H_basis
    )
    # This saves H in the new basis

    scipy.sparse.save_npz(
        "F_" + str(N) + "_allJ_Sz_" + str(2 * Sz - N) + "subspace.npz", F.transpose()
    )
    # This saves F^\dagger to be able to transform basis vectors


def saveHamiltonians(N, rampdir, magneticorder="AFM"):
    """
    The function saves the target and initial Hamiltonians for N qubits given a value N as .npz files

    :param N: length of original antiferromagnetic chain
    :param rampdir: string 'forward' or 'backward' -> allows to either do a ramp from (J_N-1 = 0 to J_N-1 = J
    and J_1 = J to J_1 = 0 <=> 'forward') or (J_1 = 0 to J_1 = J and J_N-1 = J to J_N-1 = 0 <=> 'backward')

    :return: None
    """

    # Note - requires precalculated V matrix
    if magneticorder == "AFM":
        V = scipy.sparse.load_npz("V_" + str(N) + "_allJ_Sz_1subspace.npz")

    elif magneticorder == "FM":
        V = scipy.sparse.load_npz(
            "V_" + str(N) + "_allJ_Sz_" + str(-(N - 2)) + "subspace.npz"
        )

    if rampdir == "forward":
        # Setting Hamiltonian couplings
        couplings_fs = [1 for i in range(N - 1)]
        couplingstar_fs = [1 for i in range(N - 1)]
        couplings_fs.append(0)
        couplingstar_fs.insert(0, 0)
    elif rampdir == "backward":
        couplings_fs = [1 for i in range(N - 1)]
        couplingstar_fs = [1 for i in range(N - 1)]
        couplingstar_fs.append(0)
        couplings_fs.insert(0, 0)

    # Initialising the initial and target Hamiltonian and transforming into contracted space
    H_fs = makeHamiltonianJ(N, couplings_fs)
    Htar_fs = makeHamiltonianJ(N, couplingstar_fs)
    H = V.transpose() * H_fs * V
    Htar = V.transpose() * Htar_fs * V

    scipy.sparse.save_npz("Hinitial_" + str(N) + rampdir + magneticorder + ".npz", H)
    scipy.sparse.save_npz("Htarget_" + str(N) + rampdir + magneticorder + ".npz", Htar)

    print("Hamiltonians saved successfully.")

    return


if __name__ == "__main__":
    # Script to allow running this file from terminal below, run using python SaveFiles.py -h
    parser = argparse.ArgumentParser()
    parser.add_argument("N", help="The number of qubits in the spin chain", type=int)
    parser.add_argument(
        "Sz",
        help="The value of Sz you wish to construct the transformation matrix for",
        type=int,
    )
    parser.add_argument(
        "direction", help="The direction of the ramp (forward/backward)", type=str
    )
    parser.add_argument(
        "magneticOrder",
        help="The magnetic order of the chain you are simulating",
        type=str,
    )
    args = parser.parse_args()

    saveMatrices(args.N, args.Sz)
    saveHamiltonians(args.N, args.direction, magneticorder=args.magneticOrder)
