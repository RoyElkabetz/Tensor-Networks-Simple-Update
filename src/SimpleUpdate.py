import numpy as np
import copy as cp
import time
import ncon as ncon
from scipy import linalg
from TensorNetwork import TensorNetwork


class SimpleUpdate:
    """
    This class is an implementation of the well known Tensor Network algorithm Simple Update. This implementation
    follows the steps as described the paper:
     "Universal tensor-network algorithm for any infinite lattice (2019)" - Jahromi Saeed and Orus Roman
    DOI:	10.1103/PhysRevB.99.195105
    """
    def __init__(self, tensor_network: TensorNetwork, j_ij: list, h_k: np.float, s_i: list, s_j: list,
                 s_k: list, dts: list, d_max: np.int = 2, max_iterations: np.int = 1000,
                 convergence_error: np.float = 1e-6, log_energy: np.bool = False, print_process: np.bool = True,
                 hamiltonian: np.array = None):
        """
        The default Hamiltonian implement in this algorithm is (in pseudo Latex)
                            H = J_ij \sum_{<i,j>} S_i \cdot S_j + h_k \sum_{k} S_k
        :param tensor_network:  A TensorNetwork class object (see TensorNetwork.py)
        :param j_ij:    A list of interaction coefficients of tensor pairs. The j_ij indices corfresponds to the indices of the TensorNetwork.weights list.
        :param h_k:     The "field" constant coefficient
        :param s_i:     A list of the i spin operators for spin pair interaction. s_i[n].shape = (TensorNetwork.spin_dim, TensorNetwork.spin_dim)
        :param s_j:     A list of the j spin operators for spin pair interaction. s_j[n].shape = (TensorNetwork.spin_dim, TensorNetwork.spin_dim)
        :param s_k:     A list of the i spin operators for the Hamiltonian's field term. s_k[n].shape = (TensorNetwork.spin_dim, TensorNetwork.spin_dim)
        :param d_max:   The maximal virtual bond dimension allowed in the simulation. Used in the truncation step after time-evolution.
        :param dts:      List of time steps for the imaginary time evolution (if real) or real time evolution (if complex)
        :param max_iterations:  The maximal number of iteration performed with each time step dts[n]
        :param convergence_error:  The error between time consecutive weight vector lists at which the iterative process halts.
        :param log_energy:  compute and save the energy per site value along the iterative process.
        :param print_process:  boolean for printing the parameters along iterations
        :param hamiltonian:  np.array. If not None, the class will ignore the j_ij, s_i, s_j, h_k, s_k variables and self.hamiltonian would be the 
        computed hamiltonian on all edges. hamiltonian.shape = (TensorNetwork.spin_dim ** 2, TensorNetwork.spin_dim ** 2)
        """

        self.tensors = tensor_network.tensors
        self.weights = tensor_network.weights
        self.structure_matrix = tensor_network.structure_matrix
        self.tensor_network = tensor_network
        self.j_ij = j_ij
        self.s_i = s_i
        self.s_j = s_j
        self.h_k = h_k
        self.s_k = s_k
        self.d_max = d_max
        self.dts = dts
        self.max_iterations = max_iterations
        self.convergence_error = convergence_error
        self.dt = 0.1
        self.old_weights = None
        self.logger = {'error': [],
                       'dt': [],
                       'iteration': [],
                       'energy': [],
                       'j_ij': j_ij,
                       'h_k': h_k,
                       'd_max': d_max,
                       'max_iterations': max_iterations,
                       'convergence_error': convergence_error}
        self.log_energy = log_energy
        self.print_process = print_process
        self.converged = False
        self.hamiltonian = hamiltonian

    def run(self):
        self.simple_update()
        error = np.inf
        for dt in self.dts:
            start_time = time.time()
            self.dt = dt
            for i in range(self.max_iterations):
                self.old_weights = cp.deepcopy(self.weights)
                self.simple_update()
                if i % 2 == 0 and i > 0:
                    error = self.check_convergence()
                    elapsed = time.time() - start_time
                    self.logger['error'].append(error)
                    self.logger['dt'].append(dt)
                    self.logger['iteration'].append(i)
                    if self.log_energy:
                        energy = self.energy_per_site()
                        self.logger['energy'].append(energy)
                        if self.print_process:
                            print('| dt {:2.6f} | {:5d}/{:5d} iteration | convergence error {:3.10f} '
                                  '| energy per-site {:4.10} | time {:4.2} sec'.format(dt, i, self.max_iterations, error,
                                                                                      np.round(energy, 10), elapsed))
                    else:
                        if self.print_process:
                            print('| dt {:2.6f} | {:5d}/{:5d} iteration | convergence error {:3.10f} '
                                  '| time {:4.2} sec'.format(dt, i, self.max_iterations, error, elapsed))
                    start_time = time.time()
                    if error <= self.convergence_error and dt == self.dts[-1]:
                        self.converged = True
                        self.tensor_network.su_logger = self.logger
                        return
                    if error <= self.convergence_error:
                        break
        self.tensor_network.su_logger = self.logger
        print('Simple Update did not converged. final error is {:4.10f}'.format(error))

    def check_convergence(self):
        error = 0
        for i, (old, new) in enumerate(zip(self.old_weights, self.weights)):
            error += np.sqrt(np.sum(np.square(old - new)))
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

    def absorb_sqrt_weights(self, tensor, edges_dims):
        edges = edges_dims['edges']
        dims = edges_dims['dims']
        for i, edge in enumerate(edges):
            tensor = np.einsum(tensor, np.arange(len(tensor.shape)), np.sqrt(self.weights[edge]), [dims[i]],
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
        if self.hamiltonian is not None:
            return self.hamiltonian
        i_spin_dim = self.s_i[0].shape[0]
        j_spin_dim = self.s_j[0].shape[0]
        interaction_hamiltonian = np.zeros((i_spin_dim * j_spin_dim, i_spin_dim * j_spin_dim), dtype=complex)
        for i, _ in enumerate(self.s_i):
            interaction_hamiltonian += np.kron(self.s_i[i], self.s_j[i])
        i_field_hamiltonian = np.zeros((i_spin_dim * j_spin_dim, i_spin_dim * j_spin_dim), dtype=complex)
        j_field_hamiltonian = np.zeros((i_spin_dim * j_spin_dim, i_spin_dim * j_spin_dim), dtype=complex)
        for _, s in enumerate(self.s_k):
            i_field_hamiltonian += np.kron(s, np.eye(j_spin_dim))
            j_field_hamiltonian += np.kron(np.eye(i_spin_dim), s)
        hamiltonian = self.j_ij[ek] * interaction_hamiltonian \
                      + self.h_k * (i_field_hamiltonian / i_neighbors
                                    + j_field_hamiltonian / j_neighbors)  # (i * j, i' * j')
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
        rdm = self.tensor_pair_rdm(common_edge)   # (i, j, i', j')
        return np.einsum(rdm, [0, 1, 2, 3], operator, [0, 1, 2, 3])

    def energy_per_site(self):
        """
        returns the averged energy per-site of the given Tensor Network.
        """
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
        return np.real(energy)

    def pair_expectation_per_site(self, operator):
        """
        returns the averged per-site expectation value of the tensor-pair operator parameter.
        :param operator An np.array of shape (TensorNetwork.spin_dim, TensorNetwork.spin_dim, TensorNetwork.spin_dim, TensorNetwork.spin_dim) 
        """
        expectation = 0
        for ek, _ in enumerate(self.weights):
            expectation += self.tensor_pair_expectation(ek, operator)
        expectation /= len(self.tensors)
        return np.real(expectation)

    def expectation_per_site(self, operator):
        """
        returns the averged per-site expectation value of the operator parameter.
        :param operator An np.array of shape (TensorNetwork.spin_dim, TensorNetwork.spin_dim) 
        """
        expectation = 0
        for tensor_idx, _ in enumerate(self.tensors):
            expectation += self.tensor_expectation(tensor_idx, operator)
        return np.real(expectation) / len(self.tensors)

    def absorb_all_weights(self):
        n, m = self.structure_matrix.shape
        for tensor_idx in range(n):
            tensor = self.tensors[tensor_idx]
            edges_dims = self.get_edges(tensor_idx=tensor_idx)
            tensor = self.absorb_sqrt_weights(tensor=tensor, edges_dims=edges_dims)
            self.tensors[tensor_idx] = tensor



