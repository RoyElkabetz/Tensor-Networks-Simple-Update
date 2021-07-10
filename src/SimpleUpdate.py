import numpy as np
import copy as cp
from scipy import linalg
from TensorNetwork import TensorNetwork


def simple_update(tensor_network: TensorNetwork, dt: np.float, j_ij: np.float, h_k: np.float, s_i: list, s_j: list,
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

        # permuting the indices associated with edge ek tensors ti, tj with their 1st index
        ti = tensor_dim_permute(ti)
        tj = tensor_dim_permute(tj)

        # Group all virtual indices Em!=ek to form Pl, Pr MPS tensors
        Pl = rank_n_rank_3(ti[0])
        Pr = rank_n_rank_3(tj[0])

        # SVD decomposing of Pl, Pr to obtain Q1, R and Q2, L sub-tensors, respectively
        R, sr, Q1 = truncation_svd(Pl, [0, 1], [2], keepS='yes')
        L, sl, Q2 = truncation_svd(Pr, [0, 1], [2], keepS='yes')
        R = R.dot(np.diag(sr))
        L = L.dot(np.diag(sl))

        # RQ decomposition of Pl, Pr to obtain R, Q1 and L, Q2 sub-tensors, respectively (needs fixing)
        #R, Q1 = linalg.rq(np.reshape(Pl, [Pl.shape[0] * Pl.shape[1], Pl.shape[2]]))
        #L, Q2 = linalg.rq(np.reshape(Pr, [Pr.shape[0] * Pr.shape[1], Pr.shape[2]]))

        # reshaping R and L into rank 3 tensors with shape (physical_dim, Ek_dim, Q(1/2).shape[0])
        i_physical_dim = ti[0].shape[0]
        j_physical_dim = tj[0].shape[0]
        R = rank_2_rank_3(R, i_physical_dim)  # (i, ek, Q1) (following the dimensions)
        L = rank_2_rank_3(L, j_physical_dim)  # (j, ek, Q2)

        # Contract the ITE gate with R, L, and lambda_k to form theta tensor.
        theta = time_evolution(R,
                               L,
                               lambda_k,
                               ek,
                               dt,
                               j_ij,
                               h_k,
                               s_i,
                               s_j,
                               s_k)  # (Q1, i', j', Q2)

        # Obtain R', L', lambda'_k tensors by applying an SVD to theta
        R_tild, lambda_k_tild, L_tild = truncation_svd(theta, [0, 1], [2, 3], keepS='yes', maxEigenvalNumber=d_max)

        # reshaping R_tild and L_tild back to rank 3 tensor
        R_tild = np.reshape(R_tild, (Q1.shape[0], i_physical_dim, R_tild.shape[1]))  # (Q1, i', D')
        R_tild = np.transpose(R_tild, [1, 2, 0])  # (i', D', Q1)
        L_tild = np.reshape(L_tild, (L_tild.shape[0], j_physical_dim, Q2.shape[0]))  # (D', j', Q2)
        L_tild = np.transpose(L_tild, [1, 0, 2])  # (j', D', Q2)

        # Glue back the R', L', sub-tensors to Q1, Q2, respectively, to form updated tensors P'l, P'r.
        Pl_prime = np.einsum('ijk,kl->ijl', R_tild, Q1)
        Pr_prime = np.einsum('ijk,kl->ijl', L_tild, Q2)

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
    permutation = np.array(list(range(len(tensor[0].shape))))
    permutation[[1, tensor[2][0]]] = permutation[[tensor[2][0], 1]]
    tensor[0] = np.transpose(tensor[0], permutation)
    return tensor


def rank_n_rank_3(tensor):
    if len(tensor.shape) < 3:
        raise IndexError('Error: 00002')
    shape = np.array(tensor.shape)
    newShape = [shape[0], shape[1], np.prod(shape[2:])]
    newTensor = np.reshape(tensor, newShape)
    return newTensor


def rank_2_rank_3(tensor, physicalDimension):
    if len(tensor.shape) != 2:
        raise IndexError('Error: 00003')
    newTensor = np.reshape(tensor, [physicalDimension, int(tensor.shape[0] / physicalDimension), tensor.shape[1]])
    return newTensor


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


def time_evolution(iTensor,
                   jTensor,
                   middleWeightVector,
                   commonEdge,
                   timeStep,
                   interactionConst,
                   fieldConst,
                   iOperators,
                   jOperators,
                   fieldOperators):
    d = iOperators[0].shape[0]  # physical bond dimension
    interactionHamiltonian = np.zeros((d ** 2, d ** 2), dtype=complex)
    for i in range(len(iOperators)):
        interactionHamiltonian += np.kron(iOperators[i], jOperators[i])
    Hamiltonian = -interactionConst[commonEdge] * interactionHamiltonian - 0.25 * fieldConst * (np.kron(np.eye(d), fieldOperators) + np.kron(fieldOperators, np.eye(d)))  # 0.25 is for square lattice
    unitaryGate = np.reshape(linalg.expm(-timeStep * Hamiltonian), [d, d, d, d])
    weightMatrix = np.diag(middleWeightVector)
    A = np.einsum(iTensor, [0, 1, 2], weightMatrix, [1, 3], [0, 3, 2])           # A.shape = (d(i), Weight_Vector, Q1)
    A = np.einsum(A, [0, 1, 2], jTensor, [3, 1, 4], [2, 0, 3, 4])                # A.shape = (Q1, d(i), d(j), Q2)
    theta = np.einsum(A, [0, 1, 2, 3], unitaryGate, [1, 2, 4, 5], [0, 4, 5, 3])  # theta.shape = (Q1, d(i'), d(j'), Q2)
    return theta


def normalize_tensor(tensor):
    tensorConj = np.conj(cp.copy(tensor))
    idx = list(range(len(tensor.shape)))
    norm = np.sqrt(np.einsum(tensor, idx, tensorConj, idx))
    return norm
