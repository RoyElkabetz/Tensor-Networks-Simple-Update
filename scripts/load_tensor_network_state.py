import sys
sys.path.insert(1, "../src")

import numpy as np
import matplotlib.pyplot as plt
from TensorNetwork import TensorNetwork
import SimpleUpdate as su
import structure_matrix_generator as smg

# load the tensor network
filename = 'AFH_12x12_obc_D_4'
dir_path = '../tmp/networks'
AFH_TN = TensorNetwork(load_network=True, network_name=filename, dir_path=dir_path)

# AFH Hamiltonian interaction parameters
j_ij = [1.] * AFH_TN.structure_matrix.shape[1]

# Pauli matrices
pauli_x = np.array([[0, 1],
                    [1, 0]])
pauli_y = np.array([[0, -1j],
                    [1j, 0]])
pauli_z = np.array([[1, 0],
                    [0, -1]])
s_i = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
s_j = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
s_k = [pauli_x / 2.]

# set Simple Update algorithm environment
AFH_TN_su = su.SimpleUpdate(tensor_network=AFH_TN, dts=[], j_ij=j_ij, h_k=0., s_i=s_i, s_j=s_j, s_k=s_k)
energy = AFH_TN_su.energy_per_site()
print(f'The energy per-site is: {energy}')
