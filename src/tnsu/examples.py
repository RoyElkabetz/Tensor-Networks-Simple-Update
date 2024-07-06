import numpy as np
from tnsu.utils import plot_convergence_curve
from tnsu.tensor_network import TensorNetwork
import tnsu.simple_update as su
import tnsu.structure_matrix_constructor as smg
import tnsu.math_objects as mo


def load_a_tensor_network_from_memory(network_name='AFH_10x10_obc_D_4'):
    """
    Loads a Tensor Network state from memory (given in package)
    :param network_name:
    :return:
    """
    net_names = ['AFH_10x10_obc_D_4', 'AFH_20x20_obc_D_4', 'AFH_20x20_pbc_D_4']
    assert network_name in net_names, f'There is no network "{network_name}" in memory. ' \
                                      f'Please choose from:\n {net_names}'
    dir_path = ''.join([s + '/' for s in __file__.split('/')[:-1]]) + 'networks'
    return load_a_tensor_network_state(network_name, dir_path)


def load_a_tensor_network_state(filename, dir_path):
    """
    Load an Antiferromagnetic Heisenberg PEPS ground state Tensor Network and computing its energy.
    :return: None
    """
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

    return afh_tn


def afh_peps_ground_state_experiment(size: int = 4, bc: str = 'obc', d_max_: list = [3], error: float = 1e-6,
                                     max_iterations: int = 200, dts: list = [0.1, 0.01, 0.001, 0.0001, 0.00001],
                                     h_k: float = 0., dir_path: str = '../tmp/networks',
                                     plot_results: bool = True, save_network: bool = False):
    """
    1. For every d in d_max construct a random Tensor Network state with spin dimension 2.
    2. Run Simple Update with an Antiferromagnetic Heisenberg Hamiltonian to find the Tensor Network ground state.
    3. Plot results.
    4. Return a list of networks and their energies according to the d_max list

    :param size: height and width of the Tensor Network.
    :param bc: the Tensor Network boundary condition ('pbc', 'obc') = ('open bc', 'periodic bc').
    :param d_max_: A list of maximal virtual bond dimensions, one for each experiment
    :param error: The maximally allowed L2 norm convergence error (between two consecutive weight vectors)
    :param max_iterations: Maximal number of Simple update iterations per dt
    :param dts: List of dt values for ITE
    :param h_k: The Hamiltonian's magnetic field constant (for single spin energy)
    :param dir_path: Path for saving the networks in
    :param plot_results: Bool for plot resulst, True = plot, False = don't plot
    :param save_network: Bool for saving the resulting networks
    :return: A networks list with the resulting networks
    """
    assert bc in ['obc', 'pbc'], f'bc should be in ["obc", "pbc"], instead got {bc}'

    np.random.seed(42)
    networks = []
    energies = []

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

    # Run Simple Update
    for d_max in d_max_:
        # create Tensor Network name for saving
        network_name = 'AFH_' + str(size) + 'x' + str(size) + '_pbc_' + 'D_' + str(d_max)

        # create the Tensor Network object
        afh_tn = TensorNetwork(structure_matrix=structure_matrix,
                               virtual_dim=2,
                               network_name=network_name,
                               dir_path=dir_path)

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
        if plot_results:
            plot_convergence_curve(afh_tn_su)

        # save the tensor network
        final_network = afh_tn.get_tensor_network_state()
        if save_network:
            final_network.save_network()

        # add to networks list
        networks.append(final_network)

    return networks, energies


def tensor_network_transverse_field_ground_state_experiment(smac: np.array, j_ij: list, spin: float = 0.5,
                                                            d_max_: list = [3], error: float = 1e-6,
                                                            max_iterations: int = 200, trans_field_op: str = 'x',
                                                            dts: list = [0.1, 0.01, 0.001, 0.0001, 0.00001],
                                                            h_k: float = 0., dir_path: str = '../tmp/networks',
                                                            plot_results: bool = True, save_network: bool = False,
                                                            seed: int = 42):
    """
    1. For every d in d_max construct a random Tensor Network state with spin dimension 2.
    2. Run Simple Update with the J_ij S_i\cdot S_j interactions Hamiltonian to find the Tensor Network ground state.
    3. Plot results.
    4. Return a list of networks and their energies according to the d_max list

    :param smac: the stracture matrix of the Tensor Network.
    :param j_ij: the Tensor Network J_ij interaction weights as a list in the order of the tensor network's edges
                as ordered in the stracture matrix second dimension (columns).
    :param spin: the physical spin of the tensor network, a half integer i.e., 0.5, 1, 1.5, 2,...
    :param d_max_: A list of maximal virtual bond dimensions, one for each experiment
    :param error: The maximally allowed L2 norm convergence error (between two consecutive weight vectors)
    :param max_iterations: Maximal number of Simple update iterations per dt
    :param trans_field_op: transverse field operator name 'x', 'y' or 'z'.
    :param dts: List of dt values for ITE
    :param h_k: The Hamiltonian's magnetic field constant (for single spin energy)
    :param dir_path: Path for saving the networks in
    :param plot_results: Bool for plot resulst, True = plot, False = don't plot
    :param save_network: Bool for saving the resulting networks
    :param seed: integer for seeding the numpy random functions
    :return: A networks list and ground state energies list
    """
    assert trans_field_op in ['x', 'y', 'z'], f"the trans_field_op variable should be 'x', 'y', or 'z', " \
                                              f"instead got {trans_field_op}."
    np.random.seed(seed)
    networks = []
    energies = []

    # get spin operators
    sx, sy, sz = mo.spin_operators(spin)

    # Construct the spin operators for the Hamiltonian
    s_i = [sx, sy, sz]
    s_j = [sx, sy, sz]
    s_k = []
    if trans_field_op == 'x':
        s_k.append(sx)
    elif trans_field_op == 'y':
        s_k.append(sy)
    elif trans_field_op == 'z':
        s_k.append(sz)

    print(f'There are {smac.shape[1]} edges, and {smac.shape[0]} tensors.')

    # Run Simple Update
    for d_max in d_max_:
        # create Tensor Network name for saving
        network_name = 'TN_transverse_' + trans_field_op + '__' + 'h_' + str(h_k) + '__' + 'D_' + str(d_max) + '__' + \
                       'spin_' + str(spin)

        # create the Tensor Network object
        tn = TensorNetwork(structure_matrix=smac,
                           virtual_dim=2,
                           spin_dim=int(2 * spin + 1),
                           network_name=network_name,
                           dir_path=dir_path)

        # create the Simple Update environment
        tn_su = su.SimpleUpdate(tensor_network=tn,
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
        tn_su.run()

        # compute the energy per-site observable
        energy = tn_su.energy_per_site()
        print(f'| D max: {d_max} | Energy: {energy}\n')
        energies.append(energy)

        # plot su convergence / energy curve
        if plot_results:
            plot_convergence_curve(tn_su)

        # save the tensor network
        if save_network:
            tn.save_network()

        # add to networks list
        networks.append(tn)

    return networks, energies


