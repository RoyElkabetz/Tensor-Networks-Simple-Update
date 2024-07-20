from tnsu.math_objects import spin_operators
import tnsu.simple_update as su
import tnsu.structure_matrix_constructor as smc
from tnsu.examples import (
    load_a_tensor_network_from_memory,
    afh_chain_spin_half_ground_state_experiment,
    afh_star_spin_half_ground_state_experiment,
    afh_cubic_spin_half_ground_state_experiment,
    fhf_pyrochlore_spin_half_ground_state_experiment,
    transverse_ising_field_ground_state_experiment,
)
import numpy as np

np.random.seed(42)


def test_load_tensor_network():
    # Load Tensor Network
    structure_matrix = smc.rectangular_peps_obc(height=10, width=10)
    afh_tn = load_a_tensor_network_from_memory(structure_matrix=structure_matrix, network_name="AFH_10x10_obc_D_4")
    spin = 1 / 2

    # AFH Hamiltonian interaction parameters
    j_ij = [1.0] * afh_tn.structure_matrix.shape[1]

    # get spin operators
    sx, sy, sz = spin_operators(spin)
    s_i = [sx, sy, sz]
    s_j = [sx, sy, sz]
    s_k = [sx]

    # compute energy
    afh_tn_su = su.SimpleUpdate(tensor_network=afh_tn, dts=[], j_ij=j_ij, h_k=0.0, s_i=s_i, s_j=s_j, s_k=s_k)
    energy = afh_tn_su.energy_per_site()
    assert abs(energy - -0.5616206254235453) < 1e-13


def test_afh_chain_spin_half_ground_state_experiment():
    _, energies = afh_chain_spin_half_ground_state_experiment(d_max_=[10], plot_results=False)
    assert abs(energies[0] - -0.44304) < 2e-4


def test_afh_star_ground_state_experiment():
    _, energies = afh_star_spin_half_ground_state_experiment(d_max_=[3], plot_results=False)
    assert abs(energies[0] - -0.472631) < 1e-6


def test_afh_cubic_ground_state_experiment():
    _, energies = afh_cubic_spin_half_ground_state_experiment(d_max_=[2], plot_results=False)
    assert abs(energies[0] - -0.89253) < 3e-2


def test_fhf_pyrochlore_spin_half_ground_state_experiment():
    _, energies = fhf_pyrochlore_spin_half_ground_state_experiment(
        d_max_=[3], h_k=0.1, transverse_field_op="z", plot_results=False
    )
    assert abs(energies[0] - -0.80000) < 2e-3


def test_transverse_ising_field_ground_state_experiment():
    smat = smc.infinite_structure_matrix_dict("peps")
    _, energies = transverse_ising_field_ground_state_experiment(
        smat=smat, d_max_=[2], plot_results=False, trans_field_op="x", h_k=-4.0
    )
    assert abs(energies[0] - -4.1276383) < 1e-7


def test_spin_operators():
    """
    Testing the spin_operator generation function
    """
    sx_1_2, sy_1_2, sz_1_2 = spin_operators(0.5)
    sx_1, sy_1, sz_1 = spin_operators(1.0)

    sx_1_2_expected = np.array([[0, 0.5], [0.5, 0.0]])
    sy_1_2_expected = np.array([[0, -0.5j], [0.5j, 0.0]])
    sz_1_2_expected = np.array([[0.5, 0.0], [0.0, -0.5]])

    assert np.sum(np.abs(sx_1_2 - sx_1_2_expected)) == 0
    assert np.sum(np.abs(sy_1_2 - sy_1_2_expected)) == 0
    assert np.sum(np.abs(sz_1_2 - sz_1_2_expected)) == 0

    sx_1_expected = np.array([[0, 1.0, 0.0], [1.0, 0.0, 1.0], [0, 1.0, 0.0]]) / np.sqrt(2)

    sy_1_expected = np.array([[0, -1j, 0.0], [1j, 0.0, -1j], [0, 1j, 0.0]]) / np.sqrt(2)

    sz_1_expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0, 0.0, -1.0]])

    assert np.sum(np.abs(sx_1 - sx_1_expected)) < 1e-12
    assert np.sum(np.abs(sy_1 - sy_1_expected)) < 1e-12
    assert np.sum(np.abs(sz_1 - sz_1_expected)) < 1e-12


def test_structure_matrix_validation():
    fail_too_many_dimensions = np.array([[[1.0]]])
    fail_repeated_indices = np.array([[1, 2, 3, 3, 0, 0], [0, 4, 0, 2, 1, 3], [3, 0, 1, 0, 4, 2]])
    fail_higher_index = np.array([[1, 2, 3, 6, 0, 0], [0, 6, 0, 2, 1, 3], [3, 0, 1, 0, 6, 2]])
    fail_edge_for_three = np.array([[1, 2, 3, 0, 4, 0], [0, 4, 0, 2, 1, 3], [3, 0, 1, 0, 4, 2]])
    assert not smc.is_valid(fail_too_many_dimensions)
    assert not smc.is_valid(fail_repeated_indices)
    assert not smc.is_valid(fail_higher_index)
    assert not smc.is_valid(fail_edge_for_three)
    assert not smc.is_valid(np.inf)
