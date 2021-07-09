import numpy as np
from TensorNetwork import TensorNetwork


# MPS Tensor Network
mps_structure_matrix = np.array([[1, 0, 0, 2, 3],
                                 [1, 2, 0, 0, 0],
                                 [0, 1, 2, 0, 3],
                                 [0, 0, 1, 2, 0]])
n, m = mps_structure_matrix.shape
tensors = []
weights = []

for i in range(n):
    tensors.append(np.random.rand(2, 3, 3))
for j in range(m):
    if j != 4:
        weights.append(np.ones(3) / 3)
    else:
        weights.append(np.ones(4) / 4)

mps = TensorNetwork(structure_matrix=mps_structure_matrix, weights=weights, virtual_size=4, spin_dim=2)