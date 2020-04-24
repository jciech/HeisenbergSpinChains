import scipy.sparse
from MatrixMethods import makeHamiltonianJ, makeAFMSubSpace, makeSubSpace


def saveMatrices(N, Sz):
    if (N % 2) == 0:
        H_N = makeHamiltonianJ(N, [1 for i in range(N - 1)])
        V = makeSubSpace(N, (N / 2) + Sz)
        F = makeAFMSubSpace(N, (N / 2) + Sz)

        H_basis = V * H_N * V.transpose()

    else:
        H_N = makeHamiltonianJ(N, [1 for i in range(N - 1)])
        V = makeSubSpace(N, ((N - 1) / 2) + Sz)
        F = makeAFMSubSpace(N, ((N - 1) / 2) + Sz)

        H_basis = V * H_N * V.transpose()

    scipy.sparse.save_npz('V_' + str(N) + '_allJ_Sz_' + str(Sz) + 'subspace.npz', V.transpose())
    # This saves V^\dagger to be able to transform basis vectors

    scipy.sparse.save_npz('H_' + str(N) + '_allJ_Sz_' + str(Sz) + 'subspace.npz', H_basis)
    # This saves H in the new basis

    scipy.sparse.save_npz('F_' + str(N) + '_allJ_Sz_' + str(Sz) + 'subspace.npz', F.transpose())
    # This saves F^\dagger to be able to transform basis vectors


def saveHamiltonians(N, rampdir):
    """
    The function saves the target and initial Hamiltonians for N qubits given a value N as .npz files

    :param N: length of original antiferromagnetic chain
    :param rampdir: string 'forward' or 'backward' -> allows to either do a ramp from (J_N-1 = 0 to J_N-1 = J
    and J_1 = J to J_1 = 0 <=> 'forward') or (J_1 = 0 to J_1 = J and J_N-1 = J to J_N-1 = 0 <=> 'backward')

    :return: None
    """

    # Note - requires pre-calculated V matrix
    V = scipy.sparse.load_npz('V_' + str(N) + '_allJ_Sz_1subspace.npz')

    if rampdir == 'forward':
        # Setting Hamiltonian couplings
        couplings_fs = [1 for i in range(N - 1)]
        couplingstar_fs = [1 for i in range(N - 1)]
        couplings_fs.append(0)
        couplingstar_fs.insert(0, 0)
    elif rampdir == 'backward':
        couplings_fs = [1 for i in range(N - 1)]
        couplingstar_fs = [1 for i in range(N - 1)]
        couplingstar_fs.append(0)
        couplings_fs.insert(0, 0)

    # Initialising the initial and target Hamiltonian and transforming into contracted space
    H_fs = makeHamiltonianJ(N, couplings_fs)
    Htar_fs = makeHamiltonianJ(N, couplingstar_fs)
    H = V.transpose() * H_fs * V
    Htar = V.transpose() * Htar_fs * V

    scipy.sparse.save_npz('Hinitial_' + str(N) + rampdir + '.npz', H)
    scipy.sparse.save_npz('Htarget_' + str(N) + rampdir + '.npz', Htar)

    print('Hamiltonians saved successfully.')

    return

if __name__ == '__main__':
    saveMatrices(12,0)