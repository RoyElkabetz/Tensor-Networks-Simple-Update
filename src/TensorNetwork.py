import numpy as np


class TensorNetwork:
    """A Tensor-Network object. Used in the filed of Quantum Information and Quantum Computation"""
    def __init__(self, tensors: list, weights: list, structure_matrix: np.array):
        """
        :param tensors: A list of numpy arrays of dimension k + 1. The first k dimension (which potentially can be
        different for any tensor in the list) corresponds to the virtual dimension of the Tensor Network,  and the 1
         dimension correspond to the physical dimension of the Tensor Network (Spin dimension). Each array
        corresponds to a tensor in a the Tensor Network
        :param weights: A list of 1D numpy arrays corresponds to the simple update weights between the tensors of the
        Tensor Network.
        :param structure_matrix: A 2D numpy array of integers > 0, corresponds to the interconnection between the tensor of the
        Tensor Network.
        """

        assert len(structure_matrix.shape) == 2
        assert structure_matrix.shape[0] == len(tensors)
        assert structure_matrix.shape[1] == len(weights)

        # check there are not loose connections in the network
        n, m = structure_matrix.shape
        for i in range(n):
            pass

        # verify each neighboring tensors has identical interaction dimension to their shared weights
        for i in range(n):
            for j in range(m):
                k = structure_matrix[i, j]
                if k > 0:
                    assert tensors[i].shape[k] == len(weights[j])

        self.tensors = tensors
        self.weights = weights
        self.structure_matrix = structure_matrix






