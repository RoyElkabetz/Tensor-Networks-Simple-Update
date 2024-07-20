import numpy as np

from tnsu.utils import plot_convergence_curve
from tnsu.tensor_network import TensorNetwork, DEFAULT_NETWORKS_FOLDER
import tnsu.simple_update as su
import tnsu.structure_matrix_constructor as smg
import tnsu.math_objects as mo

from typing import List
from numpy import ndarray


def load_a_tensor_network_from_memory(
    structure_matrix,
    network_name="AFH_10x10_obc_D_4",
) -> TensorNetwork:
    """
    Loads a Tensor Network state from memory (given in package)
    :params structure_matrix: the loaded tensor network corresponding structure matrix
    :param network_name: The name of the network as it is saved in DEFAULT_NETWORKS_FOLDER
    :return: A Tensor Network object
    """
    net_names = ["AFH_10x10_obc_D_4", "AFH_20x20_obc_D_4", "AFH_20x20_pbc_D_4"]
    assert network_name in net_names, (
        f'There is no network "{network_name}" in memory. ' f"Please choose from: \n {net_names}."
    )
    dir_path = DEFAULT_NETWORKS_FOLDER
    return load_a_tensor_network_state(structure_matrix, network_name, dir_path)


def load_a_tensor_network_state(structure_matrix, filename, dir_path) -> TensorNetwork:
    """
    Load an Antiferromagnetic Heisenberg PEPS ground state Tensor Network and computes its energy per site.
    :return: The Tensor Network object
    """
    afh_tn = TensorNetwork(
        structure_matrix=structure_matrix,
        network_name=filename,
        dir_path=dir_path,
        tensors=None,
        weights=None,
    )
    afh_tn.load_network()

    # AFH Hamiltonian interaction parameters
    j_ij = [1.0] * afh_tn.structure_matrix.shape[1]

    # Pauli matrices
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])

    # Construct the Hamiltonian's spin operators
    s_i = [pauli_x / 2.0, pauli_y / 2.0, pauli_z / 2.0]
    s_j = [pauli_x / 2.0, pauli_y / 2.0, pauli_z / 2.0]
    s_k = [pauli_x / 2.0]

    # Set the Simple Update algorithm environment with the loaded Tensor Network
    afh_tn_su = su.SimpleUpdate(tensor_network=afh_tn, dts=[], j_ij=j_ij, h_k=0.0, s_i=s_i, s_j=s_j, s_k=s_k)
    energy = afh_tn_su.energy_per_site()
    print(f"Loading Tensor Network: {filename}.")
    print(f"The Tensor Network energy per-site (according to the AFH Hamiltonian) is: {energy}.")

    return afh_tn


def transverse_ising_field_ground_state_experiment(
    smat: np.array,
    d_max_=None,
    error: float = 1e-6,
    max_iterations: int = 200,
    trans_field_op: str = "x",
    dts=None,
    h_k: float = 0.0,
    dir_path: str = "../tmp/networks",
    plot_results: bool = True,
    save_network: bool = False,
    seed: int = 42,
    exp_name: str = "",
) -> tuple[list[TensorNetwork], list[float]]:
    """
    1. For each `d` in `d_max`, construct a random Tensor Network state with spin dimension `2`.
    2. Run Simple Update with the -\\sigma_z \\cdot \\sigma_z -h\\sigma_k interactions Hamiltonian to find the Tensor
    Network ground state.
    3. Plot the results if specified.
    4. Return a list of networks and their energies according to the `d_max` list.

    :param smat: The structure matrix of the Tensor Network.
    :param d_max_: A list of maximal virtual bond dimensions, one for each experiment.
    :param error: The maximum allowed L2 norm convergence error (between two consecutive weight vectors).
    :param max_iterations: The maximum number of Simple Update iterations per `dt`.
    :param trans_field_op: The transverse field operator name ('x', 'y', or 'z').
    :param dts: List of `dt` values for Imaginary Time Evolution (ITE).
    :param h_k: The Hamiltonian's magnetic field constant (for single spin energy).
    :param dir_path: Path for saving the networks.
    :param plot_results: Boolean indicating whether to plot the results (True to plot, False to not plot).
    :param save_network: Boolean indicating whether to save the resulting networks.
    :param seed: Integer for seeding the numpy random functions.
    :param exp_name: The name of the experiment.
    :return: A list of networks and a list of ground state energies.
    """
    if dts is None:
        dts = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    if d_max_ is None:
        d_max_ = [3]
    assert trans_field_op in ["x", "y", "z"], (
        f"the trans_field_op variable should be 'x', 'y', or 'z', " f"instead got {trans_field_op}."
    )
    np.random.seed(seed)
    networks = []
    energies = []

    # Construct the spin operators for the Hamiltonian
    # Pauli matrices
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])

    s_i = [pauli_z]
    s_j = [pauli_z]
    s_k = []

    if trans_field_op == "x":
        s_k.append(pauli_x)
    elif trans_field_op == "y":
        s_k.append(pauli_y)
    elif trans_field_op == "z":
        s_k.append(pauli_z)

    n_tensors, m_edges = smat.shape
    j_ij = [1.0] * m_edges

    print(f"There are {m_edges} edges, and {n_tensors} tensors.")

    # Run Simple Update
    for d_max in d_max_:
        # create Tensor Network name for saving
        network_name = (
            "TIF_"
            + trans_field_op
            + "_"
            + exp_name
            + "__"
            + "h_"
            + str(h_k)
            + "__"
            + "D_"
            + str(d_max)
            + "__"
            + "spin_1/2"
        )

        # create the Tensor Network object
        tn = TensorNetwork(
            weights=None,
            tensors=None,
            structure_matrix=smat,
            virtual_dim=2,
            spin_dim=int(2),
            network_name=network_name,
            dir_path=dir_path,
        )

        # create the Simple Update environment
        tn_su = su.SimpleUpdate(
            tensor_network=tn,
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
            print_process=True,
        )

        # run Simple Update algorithm over the Tensor Network state
        tn_su.run()

        # compute the energy per-site observable
        energy = tn_su.energy_per_site()
        print(f"| D max: {d_max} | Energy: {energy}\n")
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


def afh_ground_state_experiment(
    smat: np.array,
    spin: float = 0.5,
    d_max_=None,
    error: float = 1e-6,
    max_iterations: int = 200,
    dts=None,
    dir_path: str = "../tmp/networks",
    plot_results: bool = True,
    save_network: bool = False,
    seed: int = 42,
    exp_name: str = "",
) -> tuple[list[TensorNetwork], list[float]]:
    """
    1. For each value in `d_max`, construct a random Tensor Network state with spin dimension `spin`.
    2. Run Simple Update with the Hamiltonian H = S_i⋅S_j interactions to find the Tensor Network ground state.
    3. Plot the results if specified.
    4. Return a list of networks and their energies according to the `d_max` list.

    :param smat: The structure matrix of the Tensor Network.
    :param spin: The physical spin of the tensor network, a half-integer (e.g., 0.5, 1, 1.5, 2, ...).
    :param d_max_: A list of maximal virtual bond dimensions, one for each experiment.
    :param error: The maximum allowed L2 norm convergence error (between two consecutive weight vectors).
    :param max_iterations: The maximum number of Simple Update iterations per `dt`.
    :param dts: List of `dt` values for Imaginary Time Evolution (ITE).
    :param dir_path: Path for saving the networks.
    :param plot_results: Boolean indicating whether to plot the results (True to plot, False to not plot).
    :param save_network: Boolean indicating whether to save the resulting networks.
    :param seed: Integer for seeding the numpy random functions.
    :param exp_name: The name of the experiment.
    :return: A list of networks and a list of ground state energies.
    """

    if dts is None:
        dts = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    if d_max_ is None:
        d_max_ = [3]

    np.random.seed(seed)
    networks = []
    energies = []

    # Get spin operators
    sx, sy, sz = mo.spin_operators(spin)

    # Construct the spin operators for the Hamiltonian
    s_i = [sx, sy, sz]
    s_j = [sx, sy, sz]
    s_k: List[ndarray] = []

    # Get AFH J_{ij} weights and set field to zero
    n_tensors, m_edges = smat.shape
    j_ij = [1.0] * m_edges
    h_k = 0.0

    print(f"There are {m_edges} edges, and {n_tensors} tensors.")

    # Run Simple Update
    for d_max in d_max_:
        # create Tensor Network name for saving
        network_name = "TN_AFH_" + exp_name + "__" + "D_" + str(d_max) + "__" + "spin_" + str(spin)

        # create the Tensor Network object
        tn = TensorNetwork(
            weights=None,
            tensors=None,
            structure_matrix=smat,
            virtual_dim=2,
            spin_dim=int(2 * spin + 1),
            network_name=network_name,
            dir_path=dir_path,
        )

        # create the Simple Update environment
        tn_su = su.SimpleUpdate(
            tensor_network=tn,
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
            print_process=True,
        )

        # run Simple Update algorithm over the Tensor Network state
        tn_su.run()

        # compute the energy per-site observable
        energy = tn_su.energy_per_site()
        print(f"| D max: {d_max} | Energy: {energy}\n")
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


def afh_chain_spin_half_ground_state_experiment(
    d_max_=None,
    error: float = 1e-6,
    max_iterations: int = 200,
    dts=None,
    dir_path: str = "../tmp/networks",
    plot_results: bool = True,
    save_network: bool = False,
    seed: int = 42,
    exp_name: str = "",
) -> tuple[list[TensorNetwork], list[float]]:
    """
    AFH infinite chain spin half ground state experiment, see `afh_ground_state_experiment()` function docstrings
    for variables clarifications.
    """
    print("Run AFH infinite chain spin half ground state experiment...")
    exp_name = "_chain_" + exp_name
    spin = 0.5
    smat = smg.infinite_structure_matrix_dict("chain")
    return afh_ground_state_experiment(
        smat=smat,
        spin=spin,
        d_max_=d_max_,
        error=error,
        max_iterations=max_iterations,
        dts=dts,
        dir_path=dir_path,
        plot_results=plot_results,
        save_network=save_network,
        seed=seed,
        exp_name=exp_name,
    )


def afh_star_spin_half_ground_state_experiment(
    d_max_=None,
    error: float = 1e-6,
    max_iterations: int = 200,
    dts=None,
    dir_path: str = "../tmp/networks",
    plot_results: bool = True,
    save_network: bool = False,
    seed: int = 42,
    exp_name: str = "",
) -> tuple[list[TensorNetwork], list[float]]:
    """
    AFH infinite star spin half ground state experiment, see `afh_ground_state_experiment()` function docstrings
    for variables clarifications.
    """
    print("Run AFH infinite star spin half ground state experiment...")
    exp_name = "_star_" + exp_name
    spin = 0.5
    smat = smg.infinite_structure_matrix_dict("star")
    return afh_ground_state_experiment(
        smat=smat,
        spin=spin,
        d_max_=d_max_,
        error=error,
        max_iterations=max_iterations,
        dts=dts,
        dir_path=dir_path,
        plot_results=plot_results,
        save_network=save_network,
        seed=seed,
        exp_name=exp_name,
    )


def afh_cubic_spin_half_ground_state_experiment(
    d_max_=None,
    error: float = 1e-6,
    max_iterations: int = 200,
    dts=None,
    dir_path: str = "../tmp/networks",
    plot_results: bool = True,
    save_network: bool = False,
    seed: int = 42,
    exp_name: str = "",
) -> tuple[list[TensorNetwork], list[float]]:
    """
    AFH infinite cube spin half ground state experiment, see `afh_ground_state_experiment()` function docstrings
    for variables clarifications.
    """
    print("Run AFH infinite star spin half ground state experiment...")
    exp_name = "_cube_" + exp_name
    spin = 0.5
    smat = smg.infinite_structure_matrix_dict("cube")
    return afh_ground_state_experiment(
        smat=smat,
        spin=spin,
        d_max_=d_max_,
        error=error,
        max_iterations=max_iterations,
        dts=dts,
        dir_path=dir_path,
        plot_results=plot_results,
        save_network=save_network,
        seed=seed,
        exp_name=exp_name,
    )


def fhf_ground_state_experiment(
    smat: np.array,
    spin: float = 0.5,
    d_max_=None,
    error: float = 1e-6,
    transverse_field_op: str = "x",
    h_k: float = 0.0,
    max_iterations: int = 200,
    dts=None,
    dir_path: str = "../tmp/networks",
    plot_results: bool = True,
    save_network: bool = False,
    seed: int = 42,
    exp_name: str = "",
) -> tuple[list[TensorNetwork], list[float]]:
    """
    1. For each value in `d_max`, construct a random Tensor Network state with spin dimension `spin`.
    2. Run Simple Update with the Hamiltonian H = -S_i⋅S_j -hS_k interactions to find the Tensor Network ground state.
    3. Plot the results if specified.
    4. Return a list of networks and their energies according to the `d_max` list.

    :param smat: The structure matrix of the Tensor Network.
    :param spin: The physical spin of the tensor network, a half-integer (e.g., 0.5, 1, 1.5, 2, ...).
    :param d_max_: A list of maximal virtual bond dimensions, one for each experiment.
    :param error: The maximum allowed L2 norm convergence error (between two consecutive weight vectors).
    :param transverse_field_op: Transverse field operator 'x', 'y' or 'z'
    :param h_k: Transverse field amplitude
    :param max_iterations: The maximum number of Simple Update iterations per `dt`.
    :param dts: List of `dt` values for Imaginary Time Evolution (ITE).
    :param dir_path: Path for saving the networks.
    :param plot_results: Boolean indicating whether to plot the results (True to plot, False to not plot).
    :param save_network: Boolean indicating whether to save the resulting networks.
    :param seed: Integer for seeding the numpy random functions.
    :param exp_name: The name of the experiment.
    :return: A list of networks and a list of ground state energies.
    """

    if dts is None:
        dts = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    if d_max_ is None:
        d_max_ = [3]

    np.random.seed(seed)
    networks = []
    energies = []

    # Get spin operators
    sx, sy, sz = mo.spin_operators(spin)

    # Construct the spin operators for the Hamiltonian
    s_i = [sx, sy, sz]
    s_j = [sx, sy, sz]
    s_k = []

    if transverse_field_op == "x":
        s_k.append(sx)
    elif transverse_field_op == "y":
        s_k.append(sy)
    elif transverse_field_op == "z":
        s_k.append(sz)

    # Get AFH J_{ij} weights and set field to zero
    n_tensors, m_edges = smat.shape
    j_ij = [-1.0] * m_edges

    print(f"There are {m_edges} edges, and {n_tensors} tensors.")

    # Run Simple Update
    for d_max in d_max_:
        # create Tensor Network name for saving
        network_name = "TN_FHF_" + exp_name + "__" + "D_" + str(d_max) + "__" + "spin_" + str(spin)

        # create the Tensor Network object
        tn = TensorNetwork(
            weights=None,
            tensors=None,
            structure_matrix=smat,
            virtual_dim=2,
            spin_dim=int(2 * spin + 1),
            network_name=network_name,
            dir_path=dir_path,
        )

        # create the Simple Update environment
        tn_su = su.SimpleUpdate(
            tensor_network=tn,
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
            print_process=True,
        )

        # run Simple Update algorithm over the Tensor Network state
        tn_su.run()

        # compute the energy per-site observable
        energy = tn_su.energy_per_site()
        print(f"| D max: {d_max} | Energy: {energy}\n")
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


def fhf_pyrochlore_spin_half_ground_state_experiment(
    d_max_=None,
    error: float = 1e-6,
    transverse_field_op: str = "x",
    h_k: float = 0.0,
    max_iterations: int = 200,
    dts=None,
    dir_path: str = "../tmp/networks",
    plot_results: bool = True,
    save_network: bool = False,
    seed: int = 42,
    exp_name: str = "",
) -> tuple[list[TensorNetwork], list[float]]:
    """
    FHF infinite Pyrochlore spin half ground state experiment, see `fhf_ground_state_experiment()` function docstrings
    for variables clarifications.
    """
    print("Run FHF infinite Pyrochlore spin half ground state experiment...")
    exp_name = "_pyrochlore_" + exp_name
    spin = 0.5
    smat = smg.infinite_structure_matrix_dict("pyrochlore")
    return fhf_ground_state_experiment(
        smat=smat,
        spin=spin,
        d_max_=d_max_,
        error=error,
        transverse_field_op=transverse_field_op,
        h_k=h_k,
        max_iterations=max_iterations,
        dts=dts,
        dir_path=dir_path,
        plot_results=plot_results,
        save_network=save_network,
        seed=seed,
        exp_name=exp_name,
    )
