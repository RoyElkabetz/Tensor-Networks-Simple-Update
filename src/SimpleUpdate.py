import numpy as np
import copy as cp
from scipy import linalg
from TensorNetwork import TensorNetwork


def simple_update(tensor_network: TensorNetwork, dt: np.float, j_ij: list, h_k: np.float, s_i: list, s_j: list,
                  s_k: list, d_max: np.int):
    """

    :param tensor_network:
    :param dt:
    :param j_ij:
    :param h_k:
    :param s_i:
    :param s_j:
    :param s_k:
    :param d_max:
    :return:
    """
    tensors = cp.deepcopy(tensor_network.tensors)
    weights = cp.deepcopy(tensor_network.weights)
    structure_matrix = tensor_network.structure_matrix
    n, m = np.shape(structure_matrix)

    for ek in range(m):
        lambda_k = weights[ek]

        # Find tensors ti, tj and their corresponding indices connected along edge ek.
        ti, tj = get_tensors(ek, tensors, structure_matrix)

        # collect edges and remove the ek edge from both lists
        i_edges_dims = get_edges(ti['index'], structure_matrix)
        j_edges_dims = get_edges(tj['index'], structure_matrix)

        # absorb environment (lambda weights) into tensors
        ti['tensor'] = absorb_weights(ti['tensor'], i_edges_dims, weights)
        tj['tensor'] = absorb_weights(tj['tensor'], j_edges_dims, weights)

        # permuting the indices associated with edge ek tensors ti, tj with their 1st dimension (for convenience)
        ti = tensor_dim_permute(ti)
        tj = tensor_dim_permute(tj)

        # Group all virtual indices Em!=ek to form pi, pj MPS tensors
        pi = rank_n_rank_3(ti['tensor'])
        pj = rank_n_rank_3(tj['tensor'])

        # # SVD decomposing of pi, pj to obtain qi, ri and qj, rj sub-tensors, respectively
        # ri, sr, qi = truncation_svd(pi, [0, 1], [2], keepS='yes')
        # rj, sl, qj = truncation_svd(pj, [0, 1], [2], keepS='yes')
        # ri = ri.dot(np.diag(sr))
        # rj = rj.dot(np.diag(sl))

        # RQ decomposition of pi, pj to obtain ri, qi and rj, qj sub-tensors respectively.
        ri, qi = linalg.rq(np.reshape(pi, [pi.shape[0] * pi.shape[1], pi.shape[2]]))
        rj, qj = linalg.rq(np.reshape(pj, [pj.shape[0] * pj.shape[1], pj.shape[2]]))

        # reshaping ri and rj into rank 3 tensors with shape (spin_dim, ek_dim, q_(right/left).shape[0])
        i_physical_dim = ti['tensor'].shape[0]
        j_physical_dim = tj['tensor'].shape[0]
        ri = rank_2_rank_3(ri, i_physical_dim)  # (i, ek, qi)
        rj = rank_2_rank_3(rj, j_physical_dim)  # (j, ek, qj)

        # Contract the time-evolution gate with ri, rj, and lambda_k to form a theta tensor.
        theta = time_evolution(ri, rj, lambda_k, ek, dt, j_ij[ek], h_k, s_i, s_j, s_k)  # (qi, i', j', qj)

        # Obtain ri', rj', lambda'_k tensors by applying an SVD to theta
        R_tild, lambda_k_tild, L_tild = truncation_svd(theta, [0, 1], [2, 3], keepS='yes', maxEigenvalNumber=d_max)

        # reshaping R_tild and L_tild back to rank 3 tensor
        R_tild = np.reshape(R_tild, (qi.shape[0], i_physical_dim, R_tild.shape[1]))  # (qi, i', D')
        R_tild = np.transpose(R_tild, [1, 2, 0])  # (i', D', qi)
        L_tild = np.reshape(L_tild, (L_tild.shape[0], j_physical_dim, qj.shape[0]))  # (D', j', qj)
        L_tild = np.transpose(L_tild, [1, 0, 2])  # (j', D', qj)

        # Glue back the ri', rj', sub-tensors to qi, qj, respectively, to form updated tensors P'l, P'r.
        Pl_prime = np.einsum('ijk,kl->ijl', R_tild, qi)
        Pr_prime = np.einsum('ijk,kl->ijl', L_tild, qj)

        # Reshape back the P`l, P`r to the original rank-(z + 1) tensors ti, tj
        Ti_new_shape = list(ti[0].shape)
        Ti_new_shape[1] = len(lambda_k_tild)
        Tj_new_shape = list(tj[0].shape)
        Tj_new_shape[1] = len(lambda_k_tild)
        ti[0] = rank_3_rank_n(Pl_prime, Ti_new_shape)
        tj[0] = rank_3_rank_n(Pr_prime, Tj_new_shape)

        # permuting back the legs of ti and tj
        ti = tensor_dim_permute(ti)
        tj = tensor_dim_permute(tj)

        # Remove bond matrices lambda_m from virtual legs m != ek to obtain the updated tensors ti~, tj~.
        ti[0] = absorb_inverse_weights(ti[0], i_edges_dims, weights)
        tj[0] = absorb_inverse_weights(tj[0], j_edges_dims, weights)

        # Normalize and save new ti tj and lambda_k
        tensors[ti[1][0]] = ti[0] / normalize_tensor(ti[0])
        tensors[tj[1][0]] = tj[0] / normalize_tensor(tj[0])
        weights[ek] = lambda_k_tild / np.sum(lambda_k_tild)

    return tensors, weights


########################################################################################################################
#                                                                                                                      #
#                                        SIMPLE UPDATE AUXILIARY FUNCTIONS                                             #
#                                                                                                                      #
########################################################################################################################


def get_tensors(edge, tensors, structure_matrix):
    which_tensors = np.nonzero(structure_matrix[:, edge])[0]
    tensor_dim_of_edge = structure_matrix[which_tensors, edge]
    ti = {'tensor': tensors[which_tensors[0]], 'index': which_tensors[0], 'dim': tensor_dim_of_edge[0]}
    tj = {'tensor': tensors[which_tensors[1]], 'index': which_tensors[1], 'dim': tensor_dim_of_edge[1]}
    return ti, tj


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


def absorb_inverse_weights(tensor, edgesNidx, weights):
    for i in range(len(edgesNidx[0])):
        tensor = np.einsum(tensor, list(range(len(tensor.shape))),
                              weights[int(edgesNidx[0][i])] ** (-1), [int(edgesNidx[1][i])], list(range(len(tensor.shape))))
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


def rank_3_rank_n(tensor, oldShape):
    newTensor = np.reshape(tensor, oldShape)
    return newTensor


def truncation_svd(tensor, leftIdx, rightIdx, keepS=None, maxEigenvalNumber=None):
    shape = np.array(tensor.shape)
    leftDim = np.prod(shape[leftIdx])
    rightDim = np.prod(shape[rightIdx])
    if keepS is not None:
        U, S, Vh = np.linalg.svd(tensor.reshape(leftDim, rightDim), full_matrices=False)
        if maxEigenvalNumber is not None:
            U = U[:, 0:maxEigenvalNumber]
            S = S[0:maxEigenvalNumber]
            Vh = Vh[0:maxEigenvalNumber, :]
        return U, S, Vh
    else:
        U, S, Vh = np.linalg.svd(tensor.reshape(leftDim, rightDim), full_matrices=False)
        if maxEigenvalNumber is not None:
            U = U[:, 0:maxEigenvalNumber]
            S = S[0:maxEigenvalNumber]
            Vh = Vh[0:maxEigenvalNumber, :]
        U = np.einsum(U, [0, 1], np.sqrt(S), [1], [0, 1])
        Vh = np.einsum(np.sqrt(S), [0], Vh, [0, 1], [0, 1])
    return U, Vh


def time_evolution(ri, rj, lambda_k, ek, dt, j_ij, h_k, s_i, s_j, s_k):
    p = s_i[0].shape[0]  # physical bond dimension
    interaction_hamiltonian = np.zeros((p ** 2, p ** 2), dtype=complex)
    for i, _ in enumerate(s_i):
        interaction_hamiltonian += np.kron(s_i[i], s_j[i])
    hamiltonian = -j_ij * interaction_hamiltonian - 0.25 * h_k * (np.kron(np.eye(p), s_k) + np.kron(s_k, np.eye(p)))  # 0.25 is for square lattice
    unitaryGate = np.reshape(linalg.expm(-dt * hamiltonian), [p, p, p, p])
    weightMatrix = np.diag(lambda_k)
    A = np.einsum(ri, [0, 1, 2], weightMatrix, [1, 3], [0, 3, 2])           # A.shape = (p(i), Weight_Vector, Q1)
    A = np.einsum(A, [0, 1, 2], rj, [3, 1, 4], [2, 0, 3, 4])                # A.shape = (Q1, p(i), p(j), Q2)
    theta = np.einsum(A, [0, 1, 2, 3], unitaryGate, [1, 2, 4, 5], [0, 4, 5, 3])  # theta.shape = (Q1, p(i'), p(j'), Q2)
    return theta


def normalize_tensor(tensor):
    tensorConj = np.conj(cp.copy(tensor))
    idx = list(range(len(tensor.shape)))
    norm = np.sqrt(np.einsum(tensor, idx, tensorConj, idx))
    return norm
