import numpy as np
from TensorNetwork import TensorNetwork


# MPS Tensor Network
mps_stracture_matrix = np.array([[2, 0, 0, 1],
                                 [1, 2, 0, 0],
                                 [0, 1, 2, 0],
                                 [0, 0, 1, 2]])
n, m = mps_stracture_matrix.shape
tensors = []
weights = []

for i in range(n):
    tensors.append(np.random.rand(2, 3, 3))
for j in range(m):
    weights.append(np.ones(3) / 3)

mps = TensorNetwork(tensors, weights, mps_stracture_matrix)