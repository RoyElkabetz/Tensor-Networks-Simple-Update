import numpy as np
import copy as cp
import ncon
from scipy import linalg
from TensorNetwork import TensorNetwork


def simple_update(tensor_network: TensorNetwork, dt: np.complex, j_ij: list, h_k: np.float, s_i: list, s_j: list,
                  s_k: list, d_max: np.int):
    tensors = cp.deepcopy(tensor_network.tensors)
    weights = cp.deepcopy(tensor_network.weights)
    structure_matrix = tensor_network.structure_matrix
    n, m = np.shape(structure_matrix)

    for ek in range(m):
        lambda_k = weights[ek]

        # Find tensors ti, tj and their corresponding indices connected along edge ek.
        ti, tj = get_tensors(ek, tensors, structure_matrix)

        # collect edges and remove the ek edge from both lists
        i_edges_dims = get_other_edges(ti['index'], ek, structure_matrix)
        j_edges_dims = get_other_edges(tj['index'], ek, structure_matrix)

        # absorb environment (lambda weights) into tensors
        ti['tensor'] = absorb_weights(ti['tensor'], i_edges_dims, weights)
        tj['tensor'] = absorb_weights(tj['tensor'], j_edges_dims, weights)

        # permuting the indices associated with edge ek tensors ti, tj with their 1st dimension (for convenience)
        ti = tensor_dim_permute(ti)
        tj = tensor_dim_permute(tj)

        # Group all virtual indices Em!=ek to form pi, pj MPS tensors
        pi = rank_n_rank_3(ti['tensor'])
        pj = rank_n_rank_3(tj['tensor'])

        # RQ decomposition of pi, pj to obtain ri, qi and rj, qj sub-tensors respectively.
        ri, qi = linalg.rq(np.reshape(pi, [pi.shape[0] * pi.shape[1], pi.shape[2]]))
        rj, qj = linalg.rq(np.reshape(pj, [pj.shape[0] * pj.shape[1], pj.shape[2]]))

        # reshaping ri and rj into rank 3 tensors with shape (spin_dim, ek_dim, q_(right/left).shape[0])
        i_physical_dim = ti['tensor'].shape[0]
        j_physical_dim = tj['tensor'].shape[0]
        ri = rank_2_rank_3(ri, i_physical_dim)  # (i, ek, qi)
        rj = rank_2_rank_3(rj, j_physical_dim)  # (j, ek, qj)

        # Contract the time-evolution gate with ri, rj, and lambda_k to form a theta tensor.
        i_neighbors = len(i_edges_dims['edges']) + 1
        j_neighbors = len(j_edges_dims['edges']) + 1
        theta = time_evolution(ri, rj, i_neighbors, j_neighbors,  lambda_k, dt, j_ij[ek], h_k, s_i, s_j, s_k)
        # theta.shape = (qi, i'_spin_dim, j'_spin_dim, qj)

        # Obtain ri', rj', lambda'_k tensors by applying an SVD to theta
        ri_tilde, lambda_k_tilde, rj_tilde = truncation_svd(theta, keep_s='yes', d_max=d_max)

        # reshaping ri_tilde and rj_tilde back to rank 3 tensor
        ri_tilde = np.reshape(ri_tilde, (qi.shape[0], i_physical_dim, ri_tilde.shape[1]))   # (qi, i'_spin_dim, d_max)
        ri_tilde = np.transpose(ri_tilde, [1, 2, 0])                                        # (i'_spin_dim, d_max, qi)
        rj_tilde = np.reshape(rj_tilde, (rj_tilde.shape[0], j_physical_dim, qj.shape[0]))   # (d_max, j'_spin_dim, qj)
        rj_tilde = np.transpose(rj_tilde, [1, 0, 2])                                        # (j'_spin_dim, d_max, qj)

        # Glue back the ri', rj', sub-tensors to qi, qj, respectively, to form updated tensors P'l, P'r.
        pi_prime = np.einsum('ijk,kl->ijl', ri_tilde, qi)
        pl_prime = np.einsum('ijk,kl->ijl', rj_tilde, qj)

        # Reshape back the pi_prime, pj_prime to the original rank-(z + 1) tensors ti, tj
        ti_new_shape = np.array(ti['tensor'].shape)
        ti_new_shape[1] = len(lambda_k_tilde)
        tj_new_shape = np.array(tj['tensor'].shape)
        tj_new_shape[1] = len(lambda_k_tilde)
        ti['tensor'] = rank_3_rank_n(pi_prime, ti_new_shape)
        tj['tensor'] = rank_3_rank_n(pl_prime, tj_new_shape)

        # permuting back the legs of ti and tj
        ti = tensor_dim_permute(ti)
        tj = tensor_dim_permute(tj)

        # Remove bond matrices lambda_m from virtual legs m != ek to obtain the updated tensors ti~, tj~.
        ti['tensor'] = absorb_inverse_weights(ti['tensor'], i_edges_dims, weights)
        tj['tensor'] = absorb_inverse_weights(tj['tensor'], j_edges_dims, weights)

        # Normalize and save new ti, tj and lambda_k
        tensors[ti['index']] = ti['tensor'] / tensor_norm(ti['tensor'])
        tensors[tj['index']] = tj['tensor'] / tensor_norm(tj['tensor'])
        weights[ek] = lambda_k_tilde / np.sum(lambda_k_tilde)

        # update tensor network class
        tensor_network.tensors = tensors
        tensor_network.weights = weights


########################################################################################################################
#                                                                                                                      #
#                                        SIMPLE UPDATE AUXILIARY FUNCTIONS                                             #
#                                                                                                                      #
########################################################################################################################


def get_tensors(edge, tensors, structure_matrix):
    which_tensors = np.nonzero(structure_matrix[:, edge])[0]
    tensor_dim_of_edge = structure_matrix[which_tensors, edge]
    ti = {'tensor': cp.copy(tensors[which_tensors[0]]), 'index': which_tensors[0], 'dim': tensor_dim_of_edge[0]}
    tj = {'tensor': cp.copy(tensors[which_tensors[1]]), 'index': which_tensors[1], 'dim': tensor_dim_of_edge[1]}
    return ti, tj


def get_other_edges(tensor_idx, edge, structure_matrix):
    tensor_edges = np.nonzero(structure_matrix[tensor_idx, :])[0]
    tensor_edges = np.delete(tensor_edges, np.where(tensor_edges == edge))
    tensor_dims = structure_matrix[tensor_idx, tensor_edges]
    return {'edges': tensor_edges, 'dims': tensor_dims}


def get_edges(tensor_idx, structure_matrix):
    tensor_edges = np.nonzero(structure_matrix[tensor_idx, :])[0]
    tensor_dims = structure_matrix[tensor_idx, tensor_edges]
    return {'edges': tensor_edges, 'dims': tensor_dims}


def absorb_weights(tensor, edges_dims, weights):
    edges = edges_dims['edges']
    dims = edges_dims['dims']
    for i, edge in enumerate(edges):
        tensor = np.einsum(tensor, np.arange(len(tensor.shape)), weights[edge], [dims[i]], np.arange(len(tensor.shape)))
    return tensor


def absorb_inverse_weights(tensor, edges_dims, weights):
    edges = edges_dims['edges']
    dims = edges_dims['dims']
    for i, edge in enumerate(edges):
        tensor = np.einsum(tensor, np.arange(len(tensor.shape)),
                           np.power(weights[edge], -1), [dims[i]], np.arange(len(tensor.shape)))
    return tensor


def tensor_dim_permute(tensor):
    permutation = np.arange(len(tensor['tensor'].shape))
    permutation[[1, tensor['dim']]] = permutation[[tensor['dim'], 1]]
    tensor['tensor'] = np.transpose(tensor['tensor'], permutation)
    return tensor


def rank_n_rank_3(tensor):
    """
    Turn array of shape (d1, d2, d3, ..., dn) to array of shape (d1, d2, d3 * ...* dn).
    If array shape is (d1, d2), the new shape would be (d1, d2, 1).
    """
    shape = np.array(tensor.shape)
    new_shape = [shape[0], shape[1]]
    if len(shape) > 2:
        new_shape.append(np.prod(shape[2:]))
    else:
        new_shape.append(1)
    new_tensor = np.reshape(tensor, new_shape)
    return new_tensor


def rank_2_rank_3(tensor, spin_dim):
    new_tensor = np.reshape(tensor, [spin_dim, tensor.shape[0] // spin_dim, tensor.shape[1]])
    return new_tensor


def rank_3_rank_n(tensor, old_shape):
    new_tensor = np.reshape(tensor, old_shape)
    return new_tensor


def truncation_svd(theta, keep_s=None, d_max=None):
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


def time_evolution(ri, rj, i_neighbors, j_neighbors, lambda_k, dt, j_ij, h_k, s_i, s_j, s_k):
    i_spin_dim = s_i[0].shape[0]
    j_spin_dim = s_j[0].shape[0]
    interaction_hamiltonian = np.zeros((i_spin_dim * j_spin_dim, i_spin_dim * j_spin_dim), dtype=complex)
    i_field_hamiltonian = np.zeros((i_spin_dim * j_spin_dim, i_spin_dim * j_spin_dim), dtype=complex)
    j_field_hamiltonian = np.zeros((i_spin_dim * j_spin_dim, i_spin_dim * j_spin_dim), dtype=complex)
    for i, _ in enumerate(s_i):
        interaction_hamiltonian += np.kron(s_i[i], s_j[i])
    for _, s in enumerate(s_k):
        i_field_hamiltonian += np.kron(s, np.eye(j_spin_dim))
        j_field_hamiltonian += np.kron(np.eye(i_spin_dim), s)
    hamiltonian = -j_ij * interaction_hamiltonian \
                  - h_k * (i_field_hamiltonian / i_neighbors + j_field_hamiltonian / j_neighbors)
    unitary_gate = np.reshape(linalg.expm(-dt * hamiltonian), (i_spin_dim, j_spin_dim, i_spin_dim, j_spin_dim))
    # unitary.shape = (i_spin_dim, j_spin_dim, i'_spin_dim, j'_spin_dim)
    weight_matrix = np.diag(lambda_k)
    theta = np.einsum(ri, [0, 1, 2], weight_matrix, [1, 3], [0, 3, 2])
    # theta.shape = (i_spin_dim, weight_dim, qi)
    theta = np.einsum(theta, [0, 1, 2], rj, [3, 1, 4], [2, 0, 3, 4])
    # theta.shape = (qi, i_spin_dim, j_spin_dim, qj)
    theta = np.einsum(theta, [0, 1, 2, 3], unitary_gate, [1, 2, 4, 5], [0, 4, 5, 3])
    # theta.shape = (qi, i'_spin_dim, j'_spin_dim, qj)
    return theta


def tensor_norm(tensor):
    idx = np.arange(len(tensor.shape))
    norm = np.sqrt(np.einsum(tensor, idx, np.conj(tensor), idx))
    return norm


########################################################################################################################
#                                                                                                                      #
#                                        SIMPLE UPDATE EXPECTATIONS                                                    #
#                                                                                                                      #
########################################################################################################################


def tensor_rdm(tensor_index, tensors, weights, structure_matrix):
    edges_dims = get_edges(tensor_index, structure_matrix)
    tensor = cp.copy(tensors[tensor_index])
    tensor = absorb_weights(tensor, edges_dims, weights)
    t_idx = np.arange(len(tensor.shape))
    t_conj_idx = np.arange(len(tensor.shape))
    t_conj_idx[0] = len(tensor.shape)
    rdm_idx = [0, t_conj_idx[0]]
    rdm = np.einsum(tensor, t_idx, np.conj(tensor), t_conj_idx, rdm_idx)
    return rdm / np.trace(rdm)


def tensor_pair_rdm(common_edge, tensors, weights, structure_matrix):
    common_weight = weights[common_edge]
    ti, tj = get_tensors(common_edge, tensors, structure_matrix)
    i_edges_dims = get_other_edges(ti['index'], common_edge, structure_matrix)
    j_edges_dims = get_other_edges(tj['index'], common_edge, structure_matrix)
    ti['tensor'] = absorb_weights(ti['tensor'], i_edges_dims, weights)
    tj['tensor'] = absorb_weights(tj['tensor'], j_edges_dims, weights)

    # set index lists for ncon tensor summation package
    t = 1000
    common_edge_idx = [t, t + 1]
    common_edge_conj_idx = [t + 2, t + 3]

    ti_idx = np.arange(len(ti['tensor'].shape))
    ti_idx[ti['dim']] = common_edge_idx[0]
    ti_idx[0] = -1      # i
    ti_conj_idx = np.arange(len(ti['tensor'].shape))
    ti_conj_idx[ti['dim']] = common_edge_conj_idx[0]
    ti_conj_idx[0] = -2     # i'

    tj_idx = np.arange(len(tj['tensor'].shape)) + len(ti['tensor'].shape)
    tj_idx[tj['dim']] = common_edge_idx[1]
    tj_idx[0] = -3      # j
    tj_conj_idx = np.arange(len(tj['tensor'].shape)) + len(ti['tensor'].shape)
    tj_conj_idx[tj['dim']] = common_edge_conj_idx[1]
    tj_conj_idx[0] = -4     # j'

    # use ncon package for tensors summation
    tensors = [ti['tensor'], np.conj(np.copy(ti['tensor'])), tj['tensor'], np.conj(np.copy(tj['tensor'])), np.diag(common_weight), np.diag(common_weight)]
    indices = [ti_idx, ti_conj_idx, tj_idx, tj_conj_idx, common_edge_idx, common_edge_conj_idx]
    rdm = ncon.ncon(tensors, indices)       # (i, i', j, j')
    rdm = np.reshape(rdm, (rdm.shape[0] * rdm.shape[1], rdm.shape[2] * rdm.shape[3]))       # (i, i', j, j')
    rdm /= np.trace(rdm)
    return rdm
