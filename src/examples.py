import numpy as np
from utils import plot_convergence_curve
from tensor_network import TensorNetwork
import simple_update as su
import structure_matrix_constructor as smg


def load_a_tensor_network_state():
    """
    Load an Antiferromagnetic Heisenberg 10x10 PEPS ground state Tensor Network with open boundary conditions (OBC)
    and bond-dimension D=4 from memory, and compute its energy.
    :return: None
    """

    filename = 'AFH_10x10_obc_D_4'
    dir_path = '../tmp/networks'
    afh_tn = TensorNetwork(network_name=filename, dir_path=dir_path)
    afh_tn.load_network()

    # AFH Hamiltonian interaction parameters
    j_ij = [1.] * afh_tn.structure_matrix.shape[1]

    # Pauli matrices
    pauli_x = np.array([[0, 1],
                        [1, 0]])
    pauli_y = np.array([[0, -1j],
                        [1j, 0]])
    pauli_z = np.array([[1, 0],
                        [0, -1]])

    # Construct the spin operators for the Hamiltonian
    s_i = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
    s_j = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
    s_k = [pauli_x / 2.]

    # Set the Simple Update algorithm environment with the loaded Tensor Network
    afh_tn_su = su.SimpleUpdate(tensor_network=afh_tn, dts=[], j_ij=j_ij, h_k=0., s_i=s_i, s_j=s_j, s_k=s_k)
    energy = afh_tn_su.energy_per_site()
    print(f'The network name is {filename}.')
    print(f'The Tensor Network energy per-site (according to the AFH Hamiltonian) is: {energy}.')


def afh_peps_ground_state_experiment(size=4, bc='obc', save_network=False):
    """
    1. Construct a random Tensor Network state.
    2. Run Simple Update with an Antiferromagnetic Heisenberg Hamiltonian to find the Tensor Network ground state.
    3. Plot results.
    :param: size: height and width of the Tensor Network.
    :param: bc: the Tensor Network boundary condition ('pbc', 'obc') = ('open bc', 'periodic bc').
    :param: save_network: if True, saves the tensor Network, else don't.
    :return: None
    """
    np.random.seed(42)

    # Pauli matrices
    pauli_x = np.array([[0, 1],
                        [1, 0]])
    pauli_y = np.array([[0, -1j],
                        [1j, 0]])
    pauli_z = np.array([[1, 0],
                        [0, -1]])

    # Construct the spin operators for the Hamiltonian
    s_i = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
    s_j = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
    s_k = [pauli_x / 2.]

    if bc == 'obc':
        structure_matrix = smg.rectangular_peps_obc(size, size)
    elif bc == 'pbc':
        structure_matrix = smg.square_peps_pbc(size)
    print(f'There are {structure_matrix.shape[1]} edges, and {structure_matrix.shape[0]} tensors.')

    # AFH Hamiltonian interaction parameters
    j_ij = [1.] * structure_matrix.shape[1]

    # maximal bond dimension
    d_max_ = [3]

    # convergence error between consecutive lambda weights vectors
    error = 1e-6

    # maximal number of SU iterations
    max_iterations = 200

    # time intervals for the ITE
    dts = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    # magnetic field weight (if 0, there is no magnetic field)
    h_k = 0.
    energies = []

    # Run Simple Update
    for d_max in d_max_:
        # create Tensor Network name for saving
        network_name = 'AFH_' + str(size) + 'x' + str(size) + '_pbc_' + 'D_' + str(d_max)

        # create the Tensor Network object
        afh_tn = TensorNetwork(structure_matrix=structure_matrix,
                               virtual_dim=2,
                               network_name=network_name,
                               dir_path='../tmp/networks')

        # create the Simple Update environment
        afh_tn_su = su.SimpleUpdate(tensor_network=afh_tn,
                                    dts=dts,
                                    j_ij=j_ij,
                                    h_k=h_k,
                                    s_i=s_i,
                                    s_j=s_j,
                                    s_k=s_k,
                                    d_max=d_max,
                                    max_iterations=max_iterations,
                                    convergence_error=error,
                                    log_energy=True,
                                    print_process=True)

        # run Simple Update algorithm over the Tensor Network state
        afh_tn_su.run()

        # compute the energy per-site observable
        energy = afh_tn_su.energy_per_site()
        print(f'| D max: {d_max} | Energy: {energy}\n')
        energies.append(energy)

        # plot su convergence / energy curve
        plot_convergence_curve(afh_tn_su)

        # save the tensor network
        if save_network:
            afh_tn.save_network()


if __name__ == '__main__':
    load_a_tensor_network_state()
    # afh_peps_ground_state_experiment()