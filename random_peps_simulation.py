import sys
sys.path.insert(1, "/Users/royelkabetz/Git/Tensor-Networks-Simple-Update/src")

import numpy as np
import matplotlib.pyplot as plt
from TensorNetwork import TensorNetwork
import SimpleUpdate as su
import structure_matrix_generator as smg


np.random.seed(216)

plt.rcParams.update({'font.size': 16,
                     "figure.facecolor": 'white',
                     "axes.facecolor": 'white',
                     "savefig.facecolor": 'white',
                     'savefig.edgecolor' : 'white',
                     'figure.edgecolor' : 'white'})


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

# The Tensor Network structure matrix
n = 4
structure_matrix = smg.peps_rectangular_open_boundary_conditions(n, n)
print(f'There are {structure_matrix.shape[1]} edges, and {structure_matrix.shape[0]} tensors')

# generate uniformly random Jij (interaction-weights) sampled from (-0.5, 0.5)
j_ij = np.random.random(structure_matrix.shape[1]) - 0.5

# maximal bond dimension
d_max_ = [2]

# convergence error between consecutive lambda weights vectors
error = 1e-6

# maximal number of SU iterations
max_iterations = 10

# time intervals for the ITE
dts = [0.1, 0.01, 0.001, 0.0001, 0.00001]

# magnetic field weight (if 0, there is no magnetic field)
h_k = 0.

energies = []


# Run
for d_max in d_max_:
    random_peps = TensorNetwork(structure_matrix=structure_matrix, virtual_dim=2)
    random_peps_su = su.SimpleUpdate(tensor_network=random_peps,
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
    random_peps_su.run()
    energy = random_peps_su.energy_per_site()
    print(f'| D max: {d_max} | Energy: {energy}\n')
    energies.append(energy)

# absorb all weight vectors into tensors
random_peps_su.absorb_all_weights()

# save the tensor network
random_peps.save_network('random_peps1')

# load the tensor network
# random_peps1 = TensorNetwork(load_network=True, network_name='random_peps1')
