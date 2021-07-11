import numpy as np
from TensorNetwork import TensorNetwork
from SimpleUpdate import simple_update


# MPS Tensor Network
mps_structure_matrix = np.array([[2, 0, 0, 1],
                                 [1, 2, 0, 0],
                                 [0, 1, 2, 0],
                                 [0, 0, 1, 2]])
n, m = mps_structure_matrix.shape
tensors = []
weights = []

for i in range(n):
    tensors.append(np.random.rand(2, 3, 3))
for j in range(m):
    weights.append(np.ones(3) / 3)

mps = TensorNetwork(structure_matrix=mps_structure_matrix, weights=weights)

# prepare simple-update parameters
pauli_x = np.array([[0, 1],
                    [1, 0]])
pauli_y = np.array([[0, -1j],
                    [1j, 0]])
pauli_z = np.array([[1, 0],
                    [0, -1]])
dt = 0.1
j_ij = [0.5] * len(weights)
h_k = 0.
s_i = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
s_j = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
s_k = [pauli_x / 2.]
d_max = 2

for _ in range(100):
    simple_update(tensor_network=mps, dt=dt, j_ij=j_ij, h_k=h_k, s_i=s_i, s_j=s_j, s_ik=s_k, d_max=d_max)

