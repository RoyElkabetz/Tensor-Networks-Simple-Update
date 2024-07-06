import numpy as np
import pickle
import os
import copy as cp

from typing import TypedDict

class _EdgesDict(TypedDict):
    edges : np.ndarray
    dims : np.ndarray

class TensorNetwork:
    """A Tensor-Network object. Used in the field of Quantum Information and Quantum Computation"""

    def __init__(self, structure_matrix: np.array = None, tensors: list = None, weights: list = None,
                 spin_dim: int = 2, virtual_dim: int = 3, dir_path='./networks',
                 network_name='tensor_network', random_init_real_loc: float = 1.,
                 random_init_real_scale: float = 1., random_init_imag_loc: float = None,
                 random_init_imag_scale: float = None):
        """
        :param structure_matrix: A 2D numpy array of integers > 0, corresponds to the interconnections between tensors
         and weights in the Tensor Network.
        :param tensors: A list of numpy arrays of dimension k + 1. The last k dimensions (which potentially can be
        different for any tensor in the list) corresponds to the virtual dimension of the Tensor Network, while the
        first dimension corresponds to the physical dimension of the Tensor Network (Spin dimension). Each array
        corresponds to a Tensor in the Tensor Network
        :param weights: A list of 1D numpy arrays corresponds to the simple update weights between the tensors of the
        Tensor Network.
        :param spin_dim: Relevant only in tensors==None. Then spin_dim is the size of the 0 dimension of all generated
        random tensors.
        :param virtual_dim: The virtual_dim is the size of all the generated weight vectors.
        :param dir_path: directory path for loading and saving networks.
        :param network_name: Name of the network. Also needed when loading a network.
        :param random_init_real_loc: Loc value for tensors' real random part values initialization, Gaussian(loc, scale).
        :param random_init_real_scale: Scale value for tensors' real part random values initialization, Gaussian(loc, scale).
        :param random_init_imag_loc: Loc value for tensors' imaginary random values initialization, 1j * Gaussian(loc, scale).
        :param random_init_imag_scale: Loc value for tensors' imaginary random values initialization, 1j * Gaussian(loc, scale).
        """

        # Handle tensors random initialization parameters
        assert (((random_init_real_loc is not None) and (random_init_real_scale is not None)) or
                ((random_init_imag_loc is not None) and (random_init_imag_scale is not None))) == True, \
            f"'real' or 'imag' loc and scale values for tensors' random initialization must be float values, " \
            f"instead got: {random_init_real_loc=}, {random_init_real_scale=}, {random_init_imag_loc=}, " \
            f"{random_init_imag_scale=}."

        real_init = random_init_real_loc is not None
        imag_init = random_init_imag_loc is not None
        if real_init:
            assert random_init_real_scale > 0, f"random_init_real_scale should be positive, " \
                                               f"instead got {random_init_real_scale=}"
        if imag_init:
            assert random_init_imag_scale > 0, f"random_init_imag_scale should be positive, " \
                                               f"instead got {random_init_imag_scale=}"

        if structure_matrix is not None:
            assert structure_matrix is not None, 'a structure matrix is required as an argument input.'
            assert (0 < spin_dim == int(spin_dim)), f'Spin dimension should be an integer larger than 0. ' \
                                                    f'Instead got {spin_dim}.'

            # verify the structure matrix is legit
            assert len(structure_matrix.shape) == 2, f'The given structure_matrix have {len(structure_matrix.shape)} ' \
                                                     f'dimensions, instead of 2.'
            n, m = structure_matrix.shape
            for i in range(n):
                row = structure_matrix[i, :]
                row = row[row > 0]
                assert len(set(row)) == len(row), f'Error in structure_matrix given. There are two different weights ' \
                                                  f'connected to the same dimension in tensor [{i}].'
            for j in range(m):
                column = structure_matrix[:, j]
                assert np.sum(column > 0) == 2, f'Weight vector [{j}] is not connected to two tensors.'

            if tensors is not None:
                assert n == len(tensors), f'Num of rows in structure_matrix is ' \
                                          f'{n}, while num of tensors is ' \
                                          f'{len(tensors)}. They should be equal, a row for each tensor.'
                # generate a list of uniform weights in case didn't get one as an input
                if weights is None:
                    weights = [0] * m
                    for j in range(m):
                        for i in range(n):
                            if structure_matrix[i, j] > 0:
                                break
                        weight_dim = tensors[i].shape[structure_matrix[i, j]]
                        weights[j] = np.ones(weight_dim, dtype=np.float) / weight_dim

            # generate a random gaussian tensors list in case didn't get one as input
            else:
                tensors = [0] * n
                for i in range(n):
                    tensor_shape = [spin_dim] + [0] * np.sum(structure_matrix[i, :] > 0)
                    for j in range(m):
                        if structure_matrix[i, j] > 0:
                            assert structure_matrix[i, j] <= len(tensor_shape) - 1, f'structure_matrix[{i}, {j}] = ' \
                                                                                    f'{structure_matrix[i, j]} while ' \
                                                                                    f'it should have been ' \
                                                                                    f'<= {len(tensor_shape) - 1}.'
                            if weights is not None:
                                tensor_shape[structure_matrix[i, j]] = len(weights[j])
                            else:
                                tensor_shape[structure_matrix[i, j]] = virtual_dim
                    if real_init and not imag_init:
                        tensors[i] = np.random.normal(
                            loc=random_init_real_loc * np.ones(tensor_shape),
                            scale=random_init_real_scale)
                    elif imag_init and not real_init:
                        tensors[i] = 1j * np.random.normal(
                            loc=random_init_imag_loc * np.ones(tensor_shape),
                            scale=random_init_imag_scale)
                    elif real_init and imag_init:
                        tensors[i] = np.random.normal(
                            loc=random_init_real_loc * np.ones(tensor_shape),
                            scale=random_init_real_scale) + 1j * np.random.normal(
                            loc=random_init_imag_loc * np.ones(tensor_shape),
                            scale=random_init_imag_scale)
                    else:
                        raise TypeError

                # generate a weights list in case didn't get one
                if weights is None:
                    weights = [0] * m
                    for j in range(m):
                        for i in range(n):
                            if structure_matrix[i, j] > 0:
                                break
                        weight_dim = tensors[i].shape[structure_matrix[i, j]]
                        weights[j] = np.ones(weight_dim) / weight_dim

            assert m == len(weights), f'Num of columns in structure_matrix is ' \
                                      f'{m}, while num of weights is ' \
                                      f'{len(weights)}. They should be equal !'

            # check the connectivity of each tensor in the generated tensor network
            for i in range(n):
                # all tensor virtual legs connected
                assert len(tensors[i].shape) - 1 == np.sum(structure_matrix[i, :] > 0), \
                    f'tensor [{i}] is connected to {len(tensors[i].shape) - 1}  ' \
                    f'weight vectors but have ' \
                    f'{np.sum(structure_matrix[i, :] > 0)} virtual dimensions.'

            # verify each neighboring tensors has identical interaction dimension to their shared weights
            for i in range(n):
                for j in range(m):
                    tensor_dim = structure_matrix[i, j]
                    if tensor_dim > 0:
                        assert tensors[i].shape[tensor_dim] == len(weights[j]), f'Dimension {tensor_dim} size of ' \
                                                                                f'Tensor [{i}] is' \
                                                                                f' {tensors[i].shape[tensor_dim]}, ' \
                                                                                f'while size of weight ' \
                                                                                f'vector [{j}] is {len(weights[j])}. ' \
                                                                                f'They should be equal !'
        self.virtual_dim = virtual_dim
        self.spin_dim = spin_dim
        self.tensors = tensors
        self.weights = weights
        self.structure_matrix = structure_matrix
        self.dir_path = dir_path
        self.network_name = network_name
        self.su_logger = None
        self.state_dict = None

    def create_state_dict(self):
        """
        Creates a state dictionary with all the Tensor Network object parameters
        :return: None
        """
        self.state_dict = {
            'tensors': self.tensors,
            'weights': self.weights,
            'structure_matrix': self.structure_matrix,
            'path': self.dir_path,
            'spin_dim': self.spin_dim,
            'virtual_size': self.virtual_dim,
            'network_name': self.network_name,
            'su_logger': self.su_logger
        }

    def unpack_state_dict(self):
        """
        Unpack a given state dictionary
        :return: None
        """
        self.tensors = self.state_dict['tensors']
        self.weights = self.state_dict['weights']
        self.structure_matrix = self.state_dict['structure_matrix']
        self.dir_path = self.state_dict['path']
        self.spin_dim = self.state_dict['spin_dim']
        self.virtual_dim = self.state_dict['virtual_size']
        self.network_name = self.state_dict['network_name']
        self.su_logger = self.state_dict['su_logger']

    def save_network(self, filename='tensor_network'):
        print('Saving a Tensor Network...')
        self.create_state_dict()
        if self.network_name is None:
            self.network_name = filename
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        with open(os.path.join(self.dir_path, self.network_name + '.pkl'), 'wb') as outfile:
            pickle.dump(self.state_dict, outfile, pickle.DEFAULT_PROTOCOL)

    def load_network(self, network_full_path=None):
        print('Loading a Tensor Network...')
        path_to_network = os.path.join(self.dir_path, self.network_name + '.pkl') if network_full_path is None \
            else network_full_path
        try:
            with open(path_to_network, 'rb') as infile:
                self.state_dict = pickle.load(infile)
                self.unpack_state_dict()
        except Exception as error:
            print(f'There was an error in loading the tensor network from path:\n {path_to_network}\n'
                  f'with the next exception:\n {error}.')
            
    def get_edges(self, tensor_idx:int) -> _EdgesDict:
        """
        Gets all edges and dimension of a tensor
        :param tensor_idx: the tensor index
        :return: dictionary of edges and dimensions
        """
        tensor_edges = np.nonzero(self.structure_matrix[tensor_idx, :])[0]
        tensor_dims = self.structure_matrix[tensor_idx, tensor_edges]
        return {'edges': tensor_edges, 'dims': tensor_dims}
    
    def absorb_weights(self, tensor:np.ndarray, edges_dims:dict) -> np.ndarray:
        """
        Absorb all local weights into a tensor
        :param tensor: the tensor
        :param edges_dims: an edge-dimension dict
        :return: the new tensor
        """
        edges = edges_dims['edges']
        dims = edges_dims['dims']
        for i, edge in enumerate(edges):
            tensor = np.einsum(tensor, np.arange(len(tensor.shape)), self.weights[edge], [dims[i]],
                               np.arange(len(tensor.shape)))
        return tensor

    def absorb_inverse_weights(self, tensor:np.ndarray, edges_dims:dict) -> np.ndarray:
        """
        Absorb all local inverse weights (weight^-1) into a tensor
        :param tensor: the tensor
        :param edges_dims: an edge-dimension dict
        :return: the new tensor
        """
        edges = edges_dims['edges']
        dims = edges_dims['dims']
        for i, edge in enumerate(edges):
            tensor = np.einsum(tensor, np.arange(len(tensor.shape)),
                               np.power(self.weights[edge], -1), [dims[i]], np.arange(len(tensor.shape)))
        return tensor

    def absorb_sqrt_weights(self, tensor, edges_dims):
        """
        Absorb all local sqrt(weights) into a tensor
        :param tensor: the tensor
        :param edges_dims: an edge-dimension dict
        :return: the new tensor
        """
        edges = edges_dims['edges']
        dims = edges_dims['dims']
        for i, edge in enumerate(edges):
            tensor = np.einsum(tensor, np.arange(len(tensor.shape)), np.sqrt(self.weights[edge]), [dims[i]],
                               np.arange(len(tensor.shape)))
        return tensor

    def absorb_sqrt_inverse_weights(self, tensor, edges_dims):
        """
        Absorb all local sqrt(weights)^(-1) into a tensor
        :param tensor: the tensor
        :param edges_dims: an edge-dimension dict
        :return: the new tensor
        """
        edges = edges_dims['edges']
        dims = edges_dims['dims']
        for i, edge in enumerate(edges):
            tensor = np.einsum(tensor, np.arange(len(tensor.shape)), np.power(np.sqrt(self.weights[edge]), -1),
                               [dims[i]], np.arange(len(tensor.shape)))
        return tensor

    def absorb_all_weights(self):
        """
        Absorbs all the sqrt(weights) into their neighboring tensors.
        :return: None
        """
        n, m = self.structure_matrix.shape
        for tensor_idx in range(n):
            tensor = self.tensors[tensor_idx]
            edges_dims = self.get_edges(tensor_idx=tensor_idx)
            tensor = self.absorb_sqrt_weights(tensor=tensor, edges_dims=edges_dims)
            self.tensors[tensor_idx] = tensor

    def absorb_all_inverse_weights(self):
        """
        Absorbs all the sqrt(weights)^(-1) into their neighboring tensors.
        :return: None
        """
        n, m = self.structure_matrix.shape
        for tensor_idx in range(n):
            tensor = self.tensors[tensor_idx]
            edges_dims = self.get_edges(tensor_idx=tensor_idx)
            tensor = self.absorb_sqrt_inverse_weights(tensor=tensor, edges_dims=edges_dims)
            self.tensors[tensor_idx] = tensor

    def copy(self) -> "TensorNetwork":
        """
        Returns a deep copy of this TensorNetwork.
        """
        return cp.deepcopy(self)

    def get_tensor_network_state(self) -> list[np.ndarray]:
        """
        Returns a copy of the tensor network with all sqrt(weights) absobed into their 
        negihboring tensors. This tensor-network is a PEPS represnting the state of the system.
        :returns: list[np.ndarray]: A list of tensors. The order 
        """
        tn = self.copy()
        tn.absorb_all_weights()
        return tn