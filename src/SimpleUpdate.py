import numpy as np
import copy as cp
import time
import ncon
from scipy import linalg
from TensorNetwork import TensorNetwork


class SimpleUpdate:
    """
    This class is an implementation of the well known Tensor Network algorithm Simple Update. This implementation
    follows the steps from the paper:
     "Universal tensor-network algorithm for any infinite lattice (2019)" - Jahromi Saeed and Orus Roman
    DOI:	10.1103/PhysRevB.99.195105
    """
    def __init__(self, tensor_network: TensorNetwork, j_ij: list, h_k: np.float, s_i: list, s_j: list,
                 s_k: list, dts: list, d_max: np.int = 2, max_iterations: np.int = 1000,
                 convergence_error: np.float = 1e-6):
        """
        The default Hamiltonian implement in this algorithm is (in pseudo Latex)
                            H = J_ij \sum_{<i,j>} S_i \cdot S_j + h_k \sum_{k} S_k
        :param tensor_network:  A TensorNetwork class object (see TensorNetwork.py)
        :param j_ij:    A list of interaction coefficients of tensor pairs
        :param h_k:     The "field" constant coefficient
        :param s_i:     A list of the i spin operators for spin pair interaction
        :param s_j:     A list of the j spin operators for spin pair interaction
        :param s_k:     A list of the i spin operators for the Hamiltonian's field term
        :param d_max:   The maximal virtual bond dimension allowed in the simulation
        :param dts:      List of time steps for the time evolution (if complex) or imaginary time evolution (if real)
        :param max_iterations:  The maximal number of iteration performed with each time step dt
        :param convergence_error:  The error between time consecutive weight lists at which the iterative process halts
        """
        self.tensors = tensor_network.tensors
        self.weights = tensor_network.weights
        self.structure_matrix = tensor_network.structure_matrix
        self.j_ij = j_ij
        self.h_k = h_k
        self.s_i = s_i
        self.s_j = s_j
        self.s_k = s_k
        self.d_max = d_max
        self.dts = dts
        self.max_iterations = max_iterations
        self.convergence_error = convergence_error
        self.dt = 0.1
        self.old_weights = None
        self.logger = {'error': [], 'dt': [], 'iteration': []}

    def run(self):
        self.simple_update()
        error = None
        for dt in self.dts:
            start_time = time.time()
            self.dt = dt
            for i in range(self.max_iterations):
                self.old_weights = cp.deepcopy(self.weights)
                self.simple_update()
                if i % 20 == 0 and i > 0:
                    error = self.check_convergence()
                    elapsed = time.time() - start_time
                    self.logger['error'].append(error)
                    self.logger['dt'].append(dt)
                    self.logger['iteration'].append(i)
                    print('| dt {:2.6f} | {:5d}/{:5d} iteration '
                          '| averaged error {:3.10f} | time {:4.2}'.format(dt, i, self.max_iterations, error, elapsed))
                    start_time = time.time()
                    if error <= self.convergence_error:
                        break
        print('Simple Update did not converged. final error is {:4.10f}'.format(error))

    def check_convergence(self):
        error = 0
        for i, (old, new) in enumerate(zip(self.old_weights, self.weights)):
            error += np.sqrt(np.square(old - new))
        return error / len(self.weights)

    def simple_update(self):
        """
        This function implement a single Simple Update sweep over all the edges in a tensor network.
        :return: None
        """
        tensors = self.tensors
        weights = self.weights
        structure_matrix = self.structure_matrix
        n, m = np.shape(structure_matrix)

        for ek in range(m):
            # get the edge weight vector.
            lambda_k = weights[ek]

            # get the ek tensor neighbors ti, tj and their corresponding indices connected along edge ek.
            ti, tj = self.get_tensors(ek)

            # collect ti, tj edges and dimensions and remove the ek edge and its dimension.
            i_edges_dims = self.get_other_edges(ti['index'], ek)
            j_edges_dims = self.get_other_edges(tj['index'], ek)

            # absorb environment (lambda weights) into tensors.
            ti['tensor'] = self.absorb_weights(ti['tensor'], i_edges_dims)
            tj['tensor'] = self.absorb_weights(tj['tensor'], j_edges_dims)

            # permuting the indices associated with edge ek tensors ti, tj with their 1st dimension (for convenience).
            ti = self.tensor_dim_permute(ti)
            tj = self.tensor_dim_permute(tj)

            # group all virtual indices em != ek to form pi, pj "mps" tensors.
            pi = self.rank_n_rank_3(ti['tensor'])
            pj = self.rank_n_rank_3(tj['tensor'])

            # perform RQ decomposition of pi, pj to obtain ri, qi and rj, qj sub-tensors respectively.
            ri, qi = linalg.rq(np.reshape(pi, [pi.shape[0] * pi.shape[1], pi.shape[2]]))
            rj, qj = linalg.rq(np.reshape(pj, [pj.shape[0] * pj.shape[1], pj.shape[2]]))

            # reshaping ri and rj into rank 3 tensors with shape (spin_dim, ek_dim, q_(right/left).shape[0]).
            i_physical_dim = ti['tensor'].shape[0]
            j_physical_dim = tj['tensor'].shape[0]
            ri = self.rank_2_rank_3(ri, i_physical_dim)  # (i, ek, qi)
            rj = self.rank_2_rank_3(rj, j_physical_dim)  # (j, ek, qj)

            # contract the time-evolution gate with ri, rj, and lambda_k to form a theta tensor.
            i_neighbors = len(i_edges_dims['edges']) + 1
            j_neighbors = len(j_edges_dims['edges']) + 1
            theta = self.time_evolution(ri, rj, i_neighbors, j_neighbors, lambda_k, ek)
            # theta.shape = (qi, i'_spin_dim, j'_spin_dim, qj)

            # obtain ri', rj', lambda'_k tensors by applying an SVD to theta.
            ri_tilde, lambda_k_tilde, rj_tilde = self.truncation_svd(theta, keep_s='yes')

            # reshaping ri_tilde and rj_tilde back to rank 3 tensor.
            ri_tilde = np.reshape(ri_tilde, (qi.shape[0], i_physical_dim, ri_tilde.shape[1]))
            # (qi, i'_spin_dim, d_max)
            ri_tilde = np.transpose(ri_tilde, [1, 2, 0])
            # (i'_spin_dim, d_max, qi)
            rj_tilde = np.reshape(rj_tilde, (rj_tilde.shape[0], j_physical_dim, qj.shape[0]))
            # (d_max, j'_spin_dim, qj)
            rj_tilde = np.transpose(rj_tilde, [1, 0, 2])
            # (j'_spin_dim, d_max, qj)

            # glue back the ri', rj', sub-tensors to qi, qj, respectively, to form updated tensors p'i, p'j.
            pi_prime = np.einsum('ijk,kl->ijl', ri_tilde, qi)
            pl_prime = np.einsum('ijk,kl->ijl', rj_tilde, qj)

            # reshape pi_prime and pj_prime to the original rank-(z + 1) tensors ti, tj.
            ti_new_shape = np.array(ti['tensor'].shape)
            ti_new_shape[1] = len(lambda_k_tilde)
            tj_new_shape = np.array(tj['tensor'].shape)
            tj_new_shape[1] = len(lambda_k_tilde)
            ti['tensor'] = self.rank_3_rank_n(pi_prime, ti_new_shape)
            tj['tensor'] = self.rank_3_rank_n(pl_prime, tj_new_shape)

            # permuting back the legs of ti and tj.
            ti = self.tensor_dim_permute(ti)
            tj = self.tensor_dim_permute(tj)

            # remove bond matrices lambda_m from virtual legs m != ek to obtain the updated ti, tj tensors.
            ti['tensor'] = self.absorb_inverse_weights(ti['tensor'], i_edges_dims)
            tj['tensor'] = self.absorb_inverse_weights(tj['tensor'], j_edges_dims)

            # normalize and save the updated ti, tj and lambda_k.
            tensors[ti['index']] = ti['tensor'] / self.tensor_norm(ti['tensor'])
            tensors[tj['index']] = tj['tensor'] / self.tensor_norm(tj['tensor'])
            weights[ek] = lambda_k_tilde / np.sum(lambda_k_tilde)

    def get_tensors(self, edge):
        which_tensors = np.nonzero(self.structure_matrix[:, edge])[0]
        tensor_dim_of_edge = self.structure_matrix[which_tensors, edge]
        ti = {'tensor': cp.copy(self.tensors[which_tensors[0]]),
              'index': which_tensors[0],
              'dim': tensor_dim_of_edge[0]}
        tj = {'tensor': cp.copy(self.tensors[which_tensors[1]]),
              'index': which_tensors[1],
              'dim': tensor_dim_of_edge[1]}
        return ti, tj

    def get_other_edges(self, tensor_idx, edge):
        tensor_edges = np.nonzero(self.structure_matrix[tensor_idx, :])[0]
        tensor_edges = np.delete(tensor_edges, np.where(tensor_edges == edge))
        tensor_dims = self.structure_matrix[tensor_idx, tensor_edges]
        return {'edges': tensor_edges, 'dims': tensor_dims}

    def get_edges(self, tensor_idx):
        tensor_edges = np.nonzero(self.structure_matrix[tensor_idx, :])[0]
        tensor_dims = self.structure_matrix[tensor_idx, tensor_edges]
        return {'edges': tensor_edges, 'dims': tensor_dims}

    def absorb_weights(self, tensor, edges_dims):
        edges = edges_dims['edges']
        dims = edges_dims['dims']
        for i, edge in enumerate(edges):
            tensor = np.einsum(tensor, np.arange(len(tensor.shape)), self.weights[edge], [dims[i]],
                               np.arange(len(tensor.shape)))
        return tensor

    def absorb_inverse_weights(self, tensor, edges_dims):
        edges = edges_dims['edges']
        dims = edges_dims['dims']
        for i, edge in enumerate(edges):
            tensor = np.einsum(tensor, np.arange(len(tensor.shape)),
                               np.power(self.weights[edge], -1), [dims[i]], np.arange(len(tensor.shape)))
        return tensor

    def tensor_dim_permute(self, tensor):
        permutation = np.arange(len(tensor['tensor'].shape))
        permutation[[1, tensor['dim']]] = permutation[[tensor['dim'], 1]]
        tensor['tensor'] = np.transpose(tensor['tensor'], permutation)
        return tensor

    def rank_n_rank_3(self, tensor):
        """
        Turn array of shape (d1, d2, d3, ..., dn) to array of shape (d1, d2, d3 * ...* dn).
        If array shape is (d1, d2), the new shape would be (d1, d2, 1).
        """
        shape = np.array(tensor.shape)
        new_shape = [shape[0], shape[1]]
        if len(shape) > 2:
            new_shape.append(np.prod(shape[2:]))
        elif len(shape) == 2:
            new_shape.append(1)
        else:
            raise ValueError
        new_tensor = np.reshape(tensor, new_shape)
        return new_tensor

    def rank_2_rank_3(self, tensor, spin_dim):
        new_tensor = np.reshape(tensor, [spin_dim, tensor.shape[0] // spin_dim, tensor.shape[1]])
        return new_tensor

    def rank_3_rank_n(self, tensor, old_shape):
        new_tensor = np.reshape(tensor, old_shape)
        return new_tensor

    def truncation_svd(self, theta, keep_s=None):
        d_max = self.d_max
        theta_shape = np.array(theta.shape)
        i_dim = np.prod(theta_shape[[0, 1]])
        j_dim = np.prod(theta_shape[[2, 3]])
        if keep_s is not None:
            u, s, vh = linalg.svd(theta.reshape(i_dim, j_dim), full_matrices=False)
            if d_max is not None:
                u = u[:, 0:d_max]
                s = s[0:d_max]
                vh = vh[0:d_max, :]
            return u, s, vh
        else:
            u, s, vh = np.linalg.svd(theta.reshape(i_dim, j_dim), full_matrices=False)
            if d_max is not None:
                u = u[:, 0:d_max]
                s = s[0:d_max]
                vh = vh[0:d_max, :]
            u = np.einsum(u, [0, 1], np.sqrt(s), [1], [0, 1])
            vh = np.einsum(np.sqrt(s), [0], vh, [0, 1], [0, 1])
        return u, vh

    def get_hamiltonian(self, i_neighbors, j_neighbors, ek):
        i_spin_dim = self.s_i[0].shape[0]
        j_spin_dim = self.s_j[0].shape[0]
        interaction_hamiltonian = np.zeros((i_spin_dim * j_spin_dim, i_spin_dim * j_spin_dim), dtype=complex)
        i_field_hamiltonian = np.zeros((i_spin_dim * j_spin_dim, i_spin_dim * j_spin_dim), dtype=complex)
        j_field_hamiltonian = np.zeros((i_spin_dim * j_spin_dim, i_spin_dim * j_spin_dim), dtype=complex)
        for i, _ in enumerate(self.s_i):
            interaction_hamiltonian += np.kron(self.s_i[i], self.s_j[i])
        for _, s in enumerate(self.s_k):
            i_field_hamiltonian += np.kron(s, np.eye(j_spin_dim))
            j_field_hamiltonian += np.kron(np.eye(i_spin_dim), s)
        hamiltonian = self.j_ij[ek] * interaction_hamiltonian \
                      + self.h_k * (
                                  i_field_hamiltonian / i_neighbors + j_field_hamiltonian / j_neighbors)  # (i * j, i' * j')
        return hamiltonian

    def time_evolution(self, ri, rj, i_neighbors, j_neighbors, lambda_k, ek):
        i_spin_dim = self.s_i[0].shape[0]
        j_spin_dim = self.s_j[0].shape[0]
        hamiltonian = self.get_hamiltonian(i_neighbors, j_neighbors, ek)
        unitary_gate = np.reshape(linalg.expm(-self.dt * hamiltonian), (i_spin_dim, j_spin_dim, i_spin_dim, j_spin_dim))
        # unitary.shape = (i_spin_dim, j_spin_dim, i'_spin_dim, j'_spin_dim)
        weight_matrix = np.diag(lambda_k)
        theta = np.einsum(ri, [0, 1, 2], weight_matrix, [1, 3], [0, 3, 2])
        # theta.shape = (i_spin_dim, weight_dim, qi)
        theta = np.einsum(theta, [0, 1, 2], rj, [3, 1, 4], [2, 0, 3, 4])
        # theta.shape = (qi, i_spin_dim, j_spin_dim, qj)
        theta = np.einsum(theta, [0, 1, 2, 3], unitary_gate, [1, 2, 4, 5], [0, 4, 5, 3])
        # theta.shape = (qi, i'_spin_dim, j'_spin_dim, qj)
        return theta

    def tensor_norm(self, tensor):
        idx = np.arange(len(tensor.shape))
        norm = np.sqrt(np.einsum(tensor, idx, np.conj(tensor), idx))
        return norm

    def tensor_rdm(self, tensor_index):
        edges_dims = self.get_edges(tensor_index)
        tensor = cp.copy(self.tensors[tensor_index])
        tensor = self.absorb_weights(tensor, edges_dims)
        t_idx = np.arange(len(tensor.shape))
        t_conj_idx = np.arange(len(tensor.shape))
        t_conj_idx[0] = len(tensor.shape)
        rdm_idx = [0, t_conj_idx[0]]
        rdm = np.einsum(tensor, t_idx, np.conj(tensor), t_conj_idx, rdm_idx)
        return rdm / np.trace(rdm)

    def tensor_pair_rdm(self, common_edge):
        common_weight = self.weights[common_edge]
        ti, tj = self.get_tensors(common_edge)
        i_edges_dims = self.get_other_edges(ti['index'], common_edge)
        j_edges_dims = self.get_other_edges(tj['index'], common_edge)
        ti['tensor'] = self.absorb_weights(ti['tensor'], i_edges_dims)
        tj['tensor'] = self.absorb_weights(tj['tensor'], j_edges_dims)

        # set index lists for ncon tensor summation package
        t = 1000
        common_edge_idx = [t, t + 1]
        common_edge_conj_idx = [t + 2, t + 3]

        ti_idx = np.arange(len(ti['tensor'].shape))
        ti_idx[ti['dim']] = common_edge_idx[0]
        ti_idx[0] = -1  # i
        ti_conj_idx = np.arange(len(ti['tensor'].shape))
        ti_conj_idx[ti['dim']] = common_edge_conj_idx[0]
        ti_conj_idx[0] = -3  # i'

        tj_idx = np.arange(len(tj['tensor'].shape)) + len(ti['tensor'].shape)
        tj_idx[tj['dim']] = common_edge_idx[1]
        tj_idx[0] = -2  # j
        tj_conj_idx = np.arange(len(tj['tensor'].shape)) + len(ti['tensor'].shape)
        tj_conj_idx[tj['dim']] = common_edge_conj_idx[1]
        tj_conj_idx[0] = -4  # j'

        # use ncon package for tensors summation
        tensors_list = [ti['tensor'], np.conj(np.copy(ti['tensor'])), tj['tensor'], np.conj(np.copy(tj['tensor'])),
                   np.diag(common_weight), np.diag(common_weight)]
        indices_list = [ti_idx, ti_conj_idx, tj_idx, tj_conj_idx, common_edge_idx, common_edge_conj_idx]
        rdm = ncon.ncon(tensors_list, indices_list)  # (i, j, i', j')
        rdm /= np.trace(np.reshape(rdm, (rdm.shape[0] * rdm.shape[1], rdm.shape[2] * rdm.shape[3])))
        return rdm

    def tensor_expectation(self, tensor_index, operator):
        rdm = self.tensor_rdm(tensor_index)
        return np.trace(np.matmul(rdm, operator))

    def tensor_pair_expectation(self, common_edge, operator):
        rdm = self.tensor_pair_rdm(common_edge)
        return np.einsum(rdm, [0, 1, 2, 3], operator, [0, 1, 2, 3])

    def energy_per_site(self):
        energy = 0
        i_spin_dim = self.s_i[0].shape[0]
        j_spin_dim = self.s_j[0].shape[0]
        for ek, lambda_k in enumerate(self.weights):
            ti, tj = self.get_tensors(ek)
            i_edges_dims = self.get_edges(ti['index'])
            j_edges_dims = self.get_edges(tj['index'])
            i_neighbors = len(i_edges_dims['edges'])
            j_neighbors = len(j_edges_dims['edges'])
            hamiltonian = self.get_hamiltonian(i_neighbors, j_neighbors, ek)
            hamiltonian = np.reshape(hamiltonian, (i_spin_dim, j_spin_dim, i_spin_dim, j_spin_dim))
            energy += self.tensor_pair_expectation(ek, hamiltonian)
        energy /= len(self.tensors)
        return energy

########################################################################################################################

# def simple_update(tensor_network: TensorNetwork, dt: np.complex, j_ij: list, h_k: np.float, s_i: list, s_j: list,
#                   s_k: list, d_max: np.int):
#     # tensors = cp.deepcopy(tensor_network.tensors)
#     # weights = cp.deepcopy(tensor_network.weights)
#     tensors = tensor_network.tensors
#     weights = tensor_network.weights
#     structure_matrix = tensor_network.structure_matrix
#     n, m = np.shape(structure_matrix)
#
#     for ek in range(m):
#         lambda_k = weights[ek]
#
#         # Find tensors ti, tj and their corresponding indices connected along edge ek.
#         ti, tj = get_tensors(ek, tensors, structure_matrix)
#
#         # collect edges and remove the ek edge from both lists
#         i_edges_dims = get_other_edges(ti['index'], ek, structure_matrix)
#         j_edges_dims = get_other_edges(tj['index'], ek, structure_matrix)
#
#         # absorb environment (lambda weights) into tensors
#         ti['tensor'] = absorb_weights(ti['tensor'], i_edges_dims, weights)
#         tj['tensor'] = absorb_weights(tj['tensor'], j_edges_dims, weights)
#
#         # permuting the indices associated with edge ek tensors ti, tj with their 1st dimension (for convenience)
#         ti = tensor_dim_permute(ti)
#         tj = tensor_dim_permute(tj)
#
#         # Group all virtual indices Em!=ek to form pi, pj MPS tensors
#         pi = rank_n_rank_3(ti['tensor'])
#         pj = rank_n_rank_3(tj['tensor'])
#
#         # RQ decomposition of pi, pj to obtain ri, qi and rj, qj sub-tensors respectively.
#         ri, qi = linalg.rq(np.reshape(pi, [pi.shape[0] * pi.shape[1], pi.shape[2]]))
#         rj, qj = linalg.rq(np.reshape(pj, [pj.shape[0] * pj.shape[1], pj.shape[2]]))
#
#         # reshaping ri and rj into rank 3 tensors with shape (spin_dim, ek_dim, q_(right/left).shape[0])
#         i_physical_dim = ti['tensor'].shape[0]
#         j_physical_dim = tj['tensor'].shape[0]
#         ri = rank_2_rank_3(ri, i_physical_dim)  # (i, ek, qi)
#         rj = rank_2_rank_3(rj, j_physical_dim)  # (j, ek, qj)
#
#         # Contract the time-evolution gate with ri, rj, and lambda_k to form a theta tensor.
#         i_neighbors = len(i_edges_dims['edges']) + 1
#         j_neighbors = len(j_edges_dims['edges']) + 1
#         theta = time_evolution(ri, rj, i_neighbors, j_neighbors, lambda_k, dt, j_ij[ek], h_k, s_i, s_j, s_k)
#         # theta.shape = (qi, i'_spin_dim, j'_spin_dim, qj)
#
#         # Obtain ri', rj', lambda'_k tensors by applying an SVD to theta
#         ri_tilde, lambda_k_tilde, rj_tilde = truncation_svd(theta, keep_s='yes', d_max=d_max)
#
#         # reshaping ri_tilde and rj_tilde back to rank 3 tensor
#         ri_tilde = np.reshape(ri_tilde, (qi.shape[0], i_physical_dim, ri_tilde.shape[1]))   # (qi, i'_spin_dim, d_max)
#         ri_tilde = np.transpose(ri_tilde, [1, 2, 0])                                        # (i'_spin_dim, d_max, qi)
#         rj_tilde = np.reshape(rj_tilde, (rj_tilde.shape[0], j_physical_dim, qj.shape[0]))   # (d_max, j'_spin_dim, qj)
#         rj_tilde = np.transpose(rj_tilde, [1, 0, 2])                                        # (j'_spin_dim, d_max, qj)
#
#         # Glue back the ri', rj', sub-tensors to qi, qj, respectively, to form updated tensors P'l, P'r.
#         pi_prime = np.einsum('ijk,kl->ijl', ri_tilde, qi)
#         pl_prime = np.einsum('ijk,kl->ijl', rj_tilde, qj)
#
#         # Reshape back the pi_prime, pj_prime to the original rank-(z + 1) tensors ti, tj
#         ti_new_shape = np.array(ti['tensor'].shape)
#         ti_new_shape[1] = len(lambda_k_tilde)
#         tj_new_shape = np.array(tj['tensor'].shape)
#         tj_new_shape[1] = len(lambda_k_tilde)
#         ti['tensor'] = rank_3_rank_n(pi_prime, ti_new_shape)
#         tj['tensor'] = rank_3_rank_n(pl_prime, tj_new_shape)
#
#         # permuting back the legs of ti and tj
#         ti = tensor_dim_permute(ti)
#         tj = tensor_dim_permute(tj)
#
#         # Remove bond matrices lambda_m from virtual legs m != ek to obtain the updated tensors ti~, tj~.
#         ti['tensor'] = absorb_inverse_weights(ti['tensor'], i_edges_dims, weights)
#         tj['tensor'] = absorb_inverse_weights(tj['tensor'], j_edges_dims, weights)
#
#         # Normalize and save new ti, tj and lambda_k
#         tensors[ti['index']] = ti['tensor'] / tensor_norm(ti['tensor'])
#         tensors[tj['index']] = tj['tensor'] / tensor_norm(tj['tensor'])
#         weights[ek] = lambda_k_tilde / np.sum(lambda_k_tilde)
#
#
# ########################################################################################################################
# #                                                                                                                      #
# #                                        SIMPLE UPDATE AUXILIARY FUNCTIONS                                             #
# #                                                                                                                      #
# ########################################################################################################################
#
#
# def get_tensors(edge, tensors, structure_matrix):
#     which_tensors = np.nonzero(structure_matrix[:, edge])[0]
#     tensor_dim_of_edge = structure_matrix[which_tensors, edge]
#     ti = {'tensor': cp.copy(tensors[which_tensors[0]]), 'index': which_tensors[0], 'dim': tensor_dim_of_edge[0]}
#     tj = {'tensor': cp.copy(tensors[which_tensors[1]]), 'index': which_tensors[1], 'dim': tensor_dim_of_edge[1]}
#     return ti, tj
#
#
# def get_other_edges(tensor_idx, edge, structure_matrix):
#     tensor_edges = np.nonzero(structure_matrix[tensor_idx, :])[0]
#     tensor_edges = np.delete(tensor_edges, np.where(tensor_edges == edge))
#     tensor_dims = structure_matrix[tensor_idx, tensor_edges]
#     return {'edges': tensor_edges, 'dims': tensor_dims}
#
#
# def get_edges(tensor_idx, structure_matrix):
#     tensor_edges = np.nonzero(structure_matrix[tensor_idx, :])[0]
#     tensor_dims = structure_matrix[tensor_idx, tensor_edges]
#     return {'edges': tensor_edges, 'dims': tensor_dims}
#
#
# def absorb_weights(tensor, edges_dims, weights):
#     edges = edges_dims['edges']
#     dims = edges_dims['dims']
#     for i, edge in enumerate(edges):
#         tensor = np.einsum(tensor, np.arange(len(tensor.shape)), weights[edge], [dims[i]], np.arange(len(tensor.shape)))
#     return tensor
#
#
# def absorb_inverse_weights(tensor, edges_dims, weights):
#     edges = edges_dims['edges']
#     dims = edges_dims['dims']
#     for i, edge in enumerate(edges):
#         tensor = np.einsum(tensor, np.arange(len(tensor.shape)),
#                            np.power(weights[edge], -1), [dims[i]], np.arange(len(tensor.shape)))
#     return tensor
#
#
# def tensor_dim_permute(tensor):
#     permutation = np.arange(len(tensor['tensor'].shape))
#     permutation[[1, tensor['dim']]] = permutation[[tensor['dim'], 1]]
#     tensor['tensor'] = np.transpose(tensor['tensor'], permutation)
#     return tensor
#
#
# def rank_n_rank_3(tensor):
#     """
#     Turn array of shape (d1, d2, d3, ..., dn) to array of shape (d1, d2, d3 * ...* dn).
#     If array shape is (d1, d2), the new shape would be (d1, d2, 1).
#     """
#     shape = np.array(tensor.shape)
#     new_shape = [shape[0], shape[1]]
#     if len(shape) > 2:
#         new_shape.append(np.prod(shape[2:]))
#     else:
#         new_shape.append(1)
#     new_tensor = np.reshape(tensor, new_shape)
#     return new_tensor
#
#
# def rank_2_rank_3(tensor, spin_dim):
#     new_tensor = np.reshape(tensor, [spin_dim, tensor.shape[0] // spin_dim, tensor.shape[1]])
#     return new_tensor
#
#
# def rank_3_rank_n(tensor, old_shape):
#     new_tensor = np.reshape(tensor, old_shape)
#     return new_tensor
#
#
# def truncation_svd(theta, keep_s=None, d_max=None):
#     theta_shape = np.array(theta.shape)
#     i_dim = np.prod(theta_shape[[0, 1]])
#     j_dim = np.prod(theta_shape[[2, 3]])
#     if keep_s is not None:
#         u, s, vh = linalg.svd(theta.reshape(i_dim, j_dim), full_matrices=False)
#         if d_max is not None:
#             u = u[:, 0:d_max]
#             s = s[0:d_max]
#             vh = vh[0:d_max, :]
#         return u, s, vh
#     else:
#         u, s, vh = np.linalg.svd(theta.reshape(i_dim, j_dim), full_matrices=False)
#         if d_max is not None:
#             u = u[:, 0:d_max]
#             s = s[0:d_max]
#             vh = vh[0:d_max, :]
#         u = np.einsum(u, [0, 1], np.sqrt(s), [1], [0, 1])
#         vh = np.einsum(np.sqrt(s), [0], vh, [0, 1], [0, 1])
#     return u, vh
#
#
# def get_hamiltonian(i_neighbors, j_neighbors, j_ij, h_k, s_i, s_j, s_k):
#     i_spin_dim = s_i[0].shape[0]
#     j_spin_dim = s_j[0].shape[0]
#     interaction_hamiltonian = np.zeros((i_spin_dim * j_spin_dim, i_spin_dim * j_spin_dim), dtype=complex)
#     i_field_hamiltonian = np.zeros((i_spin_dim * j_spin_dim, i_spin_dim * j_spin_dim), dtype=complex)
#     j_field_hamiltonian = np.zeros((i_spin_dim * j_spin_dim, i_spin_dim * j_spin_dim), dtype=complex)
#     for i, _ in enumerate(s_i):
#         interaction_hamiltonian += np.kron(s_i[i], s_j[i])
#     for _, s in enumerate(s_k):
#         i_field_hamiltonian += np.kron(s, np.eye(j_spin_dim))
#         j_field_hamiltonian += np.kron(np.eye(i_spin_dim), s)
#     hamiltonian = j_ij * interaction_hamiltonian \
#                     + h_k * (i_field_hamiltonian / i_neighbors + j_field_hamiltonian / j_neighbors)  # (i * j, i' * j')
#     return hamiltonian
#
#
# def time_evolution(ri, rj, i_neighbors, j_neighbors, lambda_k, dt, j_ij, h_k, s_i, s_j, s_k):
#     i_spin_dim = s_i[0].shape[0]
#     j_spin_dim = s_j[0].shape[0]
#     hamiltonian = get_hamiltonian(i_neighbors, j_neighbors, j_ij, h_k, s_i, s_j, s_k)
#     unitary_gate = np.reshape(linalg.expm(-dt * hamiltonian), (i_spin_dim, j_spin_dim, i_spin_dim, j_spin_dim))
#     # unitary.shape = (i_spin_dim, j_spin_dim, i'_spin_dim, j'_spin_dim)
#     weight_matrix = np.diag(lambda_k)
#     theta = np.einsum(ri, [0, 1, 2], weight_matrix, [1, 3], [0, 3, 2])
#     # theta.shape = (i_spin_dim, weight_dim, qi)
#     theta = np.einsum(theta, [0, 1, 2], rj, [3, 1, 4], [2, 0, 3, 4])
#     # theta.shape = (qi, i_spin_dim, j_spin_dim, qj)
#     theta = np.einsum(theta, [0, 1, 2, 3], unitary_gate, [1, 2, 4, 5], [0, 4, 5, 3])
#     # theta.shape = (qi, i'_spin_dim, j'_spin_dim, qj)
#     return theta
#
#
# def tensor_norm(tensor):
#     idx = np.arange(len(tensor.shape))
#     norm = np.sqrt(np.einsum(tensor, idx, np.conj(tensor), idx))
#     return norm
#
#
# ########################################################################################################################
# #                                                                                                                      #
# #                                        SIMPLE UPDATE EXPECTATIONS                                                    #
# #                                                                                                                      #
# ########################################################################################################################
#
#
# def tensor_rdm(tensor_index, tensors, weights, structure_matrix):
#     edges_dims = get_edges(tensor_index, structure_matrix)
#     tensor = cp.copy(tensors[tensor_index])
#     tensor = absorb_weights(tensor, edges_dims, weights)
#     t_idx = np.arange(len(tensor.shape))
#     t_conj_idx = np.arange(len(tensor.shape))
#     t_conj_idx[0] = len(tensor.shape)
#     rdm_idx = [0, t_conj_idx[0]]
#     rdm = np.einsum(tensor, t_idx, np.conj(tensor), t_conj_idx, rdm_idx)
#     return rdm / np.trace(rdm)
#
#
# def tensor_pair_rdm(common_edge, tensors, weights, structure_matrix):
#     common_weight = weights[common_edge]
#     ti, tj = get_tensors(common_edge, tensors, structure_matrix)
#     i_edges_dims = get_other_edges(ti['index'], common_edge, structure_matrix)
#     j_edges_dims = get_other_edges(tj['index'], common_edge, structure_matrix)
#     ti['tensor'] = absorb_weights(ti['tensor'], i_edges_dims, weights)
#     tj['tensor'] = absorb_weights(tj['tensor'], j_edges_dims, weights)
#
#     # set index lists for ncon tensor summation package
#     t = 1000
#     common_edge_idx = [t, t + 1]
#     common_edge_conj_idx = [t + 2, t + 3]
#
#     ti_idx = np.arange(len(ti['tensor'].shape))
#     ti_idx[ti['dim']] = common_edge_idx[0]
#     ti_idx[0] = -1      # i
#     ti_conj_idx = np.arange(len(ti['tensor'].shape))
#     ti_conj_idx[ti['dim']] = common_edge_conj_idx[0]
#     ti_conj_idx[0] = -3     # i'
#
#     tj_idx = np.arange(len(tj['tensor'].shape)) + len(ti['tensor'].shape)
#     tj_idx[tj['dim']] = common_edge_idx[1]
#     tj_idx[0] = -2      # j
#     tj_conj_idx = np.arange(len(tj['tensor'].shape)) + len(ti['tensor'].shape)
#     tj_conj_idx[tj['dim']] = common_edge_conj_idx[1]
#     tj_conj_idx[0] = -4     # j'
#
#     # use ncon package for tensors summation
#     tensors = [ti['tensor'], np.conj(np.copy(ti['tensor'])), tj['tensor'], np.conj(np.copy(tj['tensor'])), np.diag(common_weight), np.diag(common_weight)]
#     indices = [ti_idx, ti_conj_idx, tj_idx, tj_conj_idx, common_edge_idx, common_edge_conj_idx]
#     rdm = ncon.ncon(tensors, indices)       # (i, j, i', j')
#     rdm /= np.trace(np.reshape(rdm, (rdm.shape[0] * rdm.shape[1], rdm.shape[2] * rdm.shape[3])))
#     return rdm
#
#
# def tensor_expectation(tensor_index, tensors, weights, structure_matrix, operator):
#     rdm = tensor_rdm(tensor_index, tensors, weights, structure_matrix)
#     return np.trace(np.matmul(rdm, operator))
#
#
# def tensor_pair_expectation(common_edge, tensors, weights, structure_matrix, operator):
#     rdm = tensor_pair_rdm(common_edge, tensors, weights, structure_matrix)
#     return np.einsum(rdm, [0, 1, 2, 3], operator, [0, 1, 2, 3])
#
#
# def energy_per_site(tensors, weights, structure_matrix, j_ij, h_k, s_i, s_j, s_k):
#     energy = 0
#     i_spin_dim = s_i[0].shape[0]
#     j_spin_dim = s_j[0].shape[0]
#     for ek, lambda_k in enumerate(weights):
#         ti, tj = get_tensors(ek, tensors, structure_matrix)
#         i_edges_dims = get_edges(ti['index'], structure_matrix)
#         j_edges_dims = get_edges(tj['index'], structure_matrix)
#         i_neighbors = len(i_edges_dims['edges'])
#         j_neighbors = len(j_edges_dims['edges'])
#         hamiltonian = get_hamiltonian(i_neighbors, j_neighbors, j_ij[ek], h_k, s_i, s_j, s_k)
#         hamiltonian = np.reshape(hamiltonian, (i_spin_dim, j_spin_dim, i_spin_dim, j_spin_dim))
#         energy += tensor_pair_expectation(ek, tensors, weights, structure_matrix, hamiltonian)
#     energy /= len(tensors)
#     return energy