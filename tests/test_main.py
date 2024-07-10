from tnsu.math_objects import spin_operators
from tnsu.examples import *
import numpy as np
np.random.seed(42)


def test_load_tensor_network():
    # Load Tensor Network
    afh_tn = load_a_tensor_network_from_memory(network_name='AFH_10x10_obc_D_4')
    spin = 1/2

    # AFH Hamiltonian interaction parameters
    j_ij = [1.] * afh_tn.structure_matrix.shape[1]

    # get spin operators
    sx, sy, sz = spin_operators(spin)
    s_i = [sx, sy, sz]
    s_j = [sx, sy, sz]
    s_k = [sx]

    # compute energy
    afh_tn_su = su.SimpleUpdate(tensor_network=afh_tn, dts=[], j_ij=j_ij, h_k=0., s_i=s_i, s_j=s_j, s_k=s_k)
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
        d_max_=[3], h_k=0.1, transverse_field_op='z', plot_results=False)
    assert abs(energies[0] - -0.80000) < 2e-3


def test_transverse_ising_field_ground_state_experiment():
    smat = smg.infinite_structure_matrix_dict("peps")
    _, energies = transverse_ising_field_ground_state_experiment(
        smat=smat, d_max_=[2], plot_results=False, trans_field_op='x', h_k=-4.)
    assert abs(energies[0] - -4.1276383) < 1e-7


def test_spin_operators():
    """
    Testing the spin_operator generation function
    """
    sx_1_2, sy_1_2, sz_1_2 = spin_operators(0.5)
    sx_1, sy_1, sz_1 = spin_operators(1.0)

    sx_1_2_expected = np.array(
        [[0, 0.5],
         [0.5, 0.]]
    )
    sy_1_2_expected = np.array(
        [[0, -0.5j],
         [0.5j, 0.]]
    )
    sz_1_2_expected = np.array(
        [[0.5, 0.],
         [0., -0.5]]
    )

    assert np.sum(np.abs(sx_1_2 - sx_1_2_expected)) == 0
    assert np.sum(np.abs(sy_1_2 - sy_1_2_expected)) == 0
    assert np.sum(np.abs(sz_1_2 - sz_1_2_expected)) == 0

    sx_1_expected = np.array(
        [[0, 1., 0.],
         [1., 0., 1.],
         [0, 1., 0.]]
    ) / np.sqrt(2)

    sy_1_expected = np.array(
        [[0, -1j, 0.],
         [1j, 0., -1j],
         [0, 1j, 0.]]
    ) / np.sqrt(2)

    sz_1_expected = np.array(
        [[1., 0., 0.],
         [0., 0., 0.],
         [0, 0., -1.]]
    )

    assert np.sum(np.abs(sx_1 - sx_1_expected)) < 1e-12
    assert np.sum(np.abs(sy_1 - sy_1_expected)) < 1e-12
    assert np.sum(np.abs(sz_1 - sz_1_expected)) < 1e-12