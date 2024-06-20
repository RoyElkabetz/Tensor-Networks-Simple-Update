from tnsu.structure_matrix_constructor import infinite_structure_matrix_dict
import tnsu.simple_update as su
from tnsu.math_objects import spin_operators as so
from tnsu.examples import (tensor_network_transverse_field_ground_state_experiment,
                           load_a_tensor_network_from_memory)
import numpy as np
np.random.seed(42)


def test_load_tensor_network():
    # Load Tensor Network
    afh_tn = load_a_tensor_network_from_memory(network_name='AFH_10x10_obc_D_4')
    spin = 1/2

    # AFH Hamiltonian interaction parameters
    j_ij = [1.] * afh_tn.structure_matrix.shape[1]

    # get spin operators
    sx, sy, sz = so(spin)
    s_i = [sx, sy, sz]
    s_j = [sx, sy, sz]
    s_k = [sx]

    # compute energy
    afh_tn_su = su.SimpleUpdate(tensor_network=afh_tn, dts=[], j_ij=j_ij, h_k=0., s_i=s_i, s_j=s_j, s_k=s_k)
    energy = afh_tn_su.energy_per_site()
    assert abs(energy - -0.5616206254235453) < 1e-13


def test_triangle_lattice():
    smac = infinite_structure_matrix_dict("triangle")
    j_ij = [-1] * smac.shape[-1]
    networks, energies = tensor_network_transverse_field_ground_state_experiment(smac=smac, j_ij=j_ij,
                                                                                 trans_field_op='y', h_k=-0.1,
                                                                                 spin=1, seed=42,
                                                                                 dts=[0.1, 1e-6], plot_results=False)
    assert abs(energies[0] - -3.0669041827593553) < 1e-13


def test_square_lattice():
    smac = infinite_structure_matrix_dict("peps")
    j_ij = [1] * smac.shape[-1]
    networks, energies = tensor_network_transverse_field_ground_state_experiment(smac=smac, j_ij=j_ij,
                                                                                 trans_field_op='x', h_k=0.,
                                                                                 spin=0.5, seed=42,
                                                                                 max_iterations=200,
                                                                                 dts=[0.1, 0.01, 0.001, 0.0001,
                                                                                      0.00001],
                                                                                 d_max_=[3], plot_results=False)
    assert abs(energies[0] - -0.6520999369286146) < 1e-13



