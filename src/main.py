import numpy as np
from TensorNetwork import TensorNetwork
import SimpleUpdate as su


# MPS Tensor Network
mps_structure_matrix = np.array([[1, 2],
                                 [1, 2]])
# MPS Tensor Network
peps_structure_matrix = np.array([[1, 2, 3, 4, 0, 0, 0, 0],
                                  [1, 2, 0, 0, 3, 4, 0, 0],
                                  [0, 0, 1, 2, 0, 0, 3, 4,],
                                  [0, 0, 0, 0, 1, 2, 3, 4]])

star_structure_matrix = np.array([[1, 2, 3, 0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 2, 3, 0, 0, 0, 0],
                                 [0, 1, 0, 2, 0, 3, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 2, 3, 0],
                                 [0, 0, 0, 0, 1, 0, 2, 0, 3],
                                 [0, 0, 1, 0, 0, 0, 0, 2, 3]])

cube_structure_matrix = np.array([[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 2, 0, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 5, 6]])


n, m = mps_structure_matrix.shape
tensors = []
weights = []

for i in range(n):
    tensors.append(np.random.rand(2, 3, 3))
for j in range(m):
    weights.append(np.ones(3) / 3)

# prepare simple-update parameters
pauli_x = np.array([[0, 1],
                    [1, 0]])
pauli_y = np.array([[0, -1j],
                    [1j, 0]])
pauli_z = np.array([[1, 0],
                    [0, -1]])
dts = [0.1, 0.01, 0.001, 0.0001, 0.00001]
h_k = 0.
s_i = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
s_j = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
s_k = [pauli_x / 2.]
d_max = 2


mps = TensorNetwork(structure_matrix=mps_structure_matrix, weights=weights)
peps = TensorNetwork(structure_matrix=peps_structure_matrix, virtual_size=2)
star = TensorNetwork(structure_matrix=star_structure_matrix, virtual_size=2)
cube = TensorNetwork(structure_matrix=cube_structure_matrix, virtual_size=2)
j_ij_mps = [-1.] * len(mps.weights)
j_ij_peps = [-1.] * len(peps.weights)
j_ij_star = [-1.] * len(star.weights)
j_ij_cube = [-1.] * len(cube.weights)

for dt in dts:
    for _ in range(50):
        su.simple_update(tensor_network=cube, dt=dt, j_ij=j_ij_cube, h_k=h_k, s_i=s_i, s_j=s_j, s_ik=s_k, s_jk=s_k, d_max=d_max)
tensors = cube.tensors
weights = cube.weights
cube_energy = su.energy_per_site(tensors, weights, cube_structure_matrix, j_ij_cube, h_k, s_i, s_j, s_k, s_k)

for dt in dts:
    for _ in range(50):
        su.simple_update(tensor_network=mps, dt=dt, j_ij=j_ij_mps, h_k=h_k, s_i=s_i, s_j=s_j, s_ik=s_k, s_jk=s_k, d_max=d_max)
tensors = mps.tensors
weights = mps.weights
mps_energy = su.energy_per_site(tensors, weights, mps_structure_matrix, j_ij_mps, h_k, s_i, s_j, s_k, s_k)

for dt in dts:
    for _ in range(50):
        su.simple_update(tensor_network=star, dt=dt, j_ij=j_ij_star, h_k=h_k, s_i=s_i, s_j=s_j, s_ik=s_k, s_jk=s_k, d_max=d_max)
tensors = star.tensors
weights = star.weights
star_energy = su.energy_per_site(tensors, weights, star_structure_matrix, j_ij_star, h_k, s_i, s_j, s_k, s_k)
