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

    for Ek in range(m):
        lambda_k = weights[Ek]

        # Find tensors ti, tj and their corresponding indices connected along edge Ek.
        ti, tj = get_tensors(Ek, tensors, structure_matrix)

        # collect edges and remove the Ek edge from both lists
        i_edges_dims = get_edges(ti['index'], structure_matrix)
        j_edges_dims = get_edges(tj['index'], structure_matrix)

        # absorb environment (lambda weights) into tensors
        ti[0] = absorb_weights(ti[0], i_edges_dims, weights)
        tj[0] = absorb_weights(tj[0], j_edges_dims, weights)

        # permuting the indices associated with edge Ek tensors ti, tj with their 1st index
        ti = tensor_dim_permute(ti)
        tj = tensor_dim_permute(tj)

        # Group all virtual indices Em!=Ek to form Pl, Pr MPS tensors
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
        R = rank_2_rank_3(R, i_physical_dim)  # (i, Ek, Q1) (following the dimensions)
        L = rank_2_rank_3(L, j_physical_dim)  # (j, Ek, Q2)

        # Contract the ITE gate with R, L, and lambda_k to form theta tensor.
        theta = time_evolution(R,
                               L,
                               lambda_k,
                               Ek,
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

        # Remove bond matrices lambda_m from virtual legs m != Ek to obtain the updated tensors ti~, tj~.
        ti[0] = absorb_inverse_weights(ti[0], i_edges_dims, weights)
        tj[0] = absorb_inverse_weights(tj[0], j_edges_dims, weights)

        # Normalize and save new ti tj and lambda_k
        tensors[ti[1][0]] = ti[0] / normalize_tensor(ti[0])
        tensors[tj[1][0]] = tj[0] / normalize_tensor(tj[0])
        weights[Ek] = lambda_k_tild / np.sum(lambda_k_tild)

    return tensors, weights


########################################################################################################################
#                                                                                                                      #
#                                        SIMPLE UPDATE AUXILIARY FUNCTIONS                                             #
#                                                                                                                      #
########################################################################################################################


def get_tensors(edge, tensors, structure_matrix):
    which_tensors = np.nonzero(structure_matrix[:, edge])[0]
    tensor_dim_of_edge = structure_matrix[which_tensors, edge]
    ti = {'tensor': tensors[which_tensors[0]],
          'index': which_tensors[0],
          'dim': tensor_dim_of_edge[0]
          }
    tj = {'tensor': tensors[which_tensors[1]],
          'index': which_tensors[1],
          'dim': tensor_dim_of_edge[1]
          }
    return ti, tj


# def get_edges(edge, smat):
#     """
#     Given an edge, collect neighboring tensors edges and indices
#     :param edge: edge number {0, 1, ..., m-1}.
#     :param smat: structure matrix (n x m).
#     :return: two lists of Ti, Tj edges and associated indices with 'edge' and its index removed.
#     """
#     tensorNumber = np.nonzero(smat[:, edge])[0]
#     iEdgesNidx = [list(np.nonzero(smat[tensorNumber[0], :])[0]),
#                   list(smat[tensorNumber[0], np.nonzero(smat[tensorNumber[0], :])[0]])
#                   ]  # [edges, indices]
#     jEdgesNidx = [list(np.nonzero(smat[tensorNumber[1], :])[0]),
#                   list(smat[tensorNumber[1], np.nonzero(smat[tensorNumber[1], :])[0]])
#                   ]  # [edges, indices]
#     # remove 'edge' and its associated index from both i, j lists.
#     iEdgesNidx[0].remove(edge)
#     iEdgesNidx[1].remove(smat[tensorNumber[0], edge])
#     jEdgesNidx[0].remove(edge)
#     jEdgesNidx[1].remove(smat[tensorNumber[1], edge])
#     return iEdgesNidx, jEdgesNidx


def get_edges(tensor_idx, structure_matrix):
    """
    Given an index of a tensor, return all of its tensor_edges and associated tensor_dims.
    :param tensor_idx: the tensor index in the structure matrix
    :param structure_matrix: structure matrix
    :return: list of two lists [[tensor_edges], [tensor_dims]].
    """
    tensor_edges = np.nonzero(structure_matrix[tensor_idx, :])[0]
    tensor_dims = structure_matrix[tensor_idx, tensor_edges]
    return {'edges': tensor_edges, 'dims': tensor_dims}


def getAllTensorsEdges(edge, smat):
    """
    Given an edge, collect neighboring tensors edges and indices
    :param edge: edge number {0, 1, ..., m-1}.
    :param smat: structure matrix (n x m).
    :return: two lists of Ti, Tj edges and associated indices.
    """
    tensorNumber = np.nonzero(smat[:, edge])[0]
    iEdgesNidx = [list(np.nonzero(smat[tensorNumber[0], :])[0]),
                  list(smat[tensorNumber[0], np.nonzero(smat[tensorNumber[0], :])[0]])
                  ]  # [edges, indices]
    jEdgesNidx = [list(np.nonzero(smat[tensorNumber[1], :])[0]),
                  list(smat[tensorNumber[1], np.nonzero(smat[tensorNumber[1], :])[0]])
                  ]  # [edges, indices]
    return iEdgesNidx, jEdgesNidx


def absorb_weights(tensor, edgesNidx, weights):
    """
    Absorb neighboring lambda weights into tensor.
    :param tensor: tensor
    :param edgesNidx: list of two lists [[edges], [indices]].
    :param weights: list of lambda weights.
    :return: the new tensor list [new_tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    for i in range(len(edgesNidx[0])):
        tensor = np.einsum(tensor, list(range(len(tensor.shape))), weights[int(edgesNidx[0][i])], [int(edgesNidx[1][i])], list(range(len(tensor.shape))))
    return tensor


def absorbSqrtWeights(tensor, edgesNidx, weights):
    """
    Absorb square root of neighboring lambda weights into tensor.
    :param tensor: tensor
    :param edgesNidx: list of two lists [[edges], [indices]].
    :param weights: list of lambda weights.
    :return: the new tensor list [new_tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    for i in range(len(edgesNidx[0])):
        tensor = np.einsum(tensor, list(range(len(tensor.shape))), np.sqrt(weights[int(edgesNidx[0][i])]),
                              [int(edgesNidx[1][i])], list(range(len(tensor.shape))))
    return tensor


def absorb_inverse_weights(tensor, edgesNidx, weights):
    """
    Absorb inverse neighboring lambda weights into tensor.
    :param tensor: tensor
    :param edgesNidx: list of two lists [[edges], [indices]].
    :param weights: list of lambda weights.
    :return: the new tensor list [new_tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    for i in range(len(edgesNidx[0])):
        tensor = np.einsum(tensor, list(range(len(tensor.shape))),
                              weights[int(edgesNidx[0][i])] ** (-1), [int(edgesNidx[1][i])], list(range(len(tensor.shape))))
    return tensor


def tensor_dim_permute(tensor):
    """
    Swapping the 'tensor_index_along_edge' index with the 1st index
    :param tensor: [tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    :return: the list with the permuted tensor [permuted_tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    permutation = np.array(list(range(len(tensor[0].shape))))
    permutation[[1, tensor[2][0]]] = permutation[[tensor[2][0], 1]]
    tensor[0] = np.transpose(tensor[0], permutation)
    return tensor


def rank_n_rank_3(tensor):
    """
    Taking a rank-N tensor (N >= 3) and make it a rank-3 tensor by grouping all indices (2, 3, ..., N - 1).
    :param tensor: the tensor
    :return: the reshaped rank-3 tensor.
    """
    if len(tensor.shape) < 3:
        raise IndexError('Error: 00002')
    shape = np.array(tensor.shape)
    newShape = [shape[0], shape[1], np.prod(shape[2:])]
    newTensor = np.reshape(tensor, newShape)
    return newTensor


def rank_2_rank_3(tensor, physicalDimension):
    """
    Taking a rank-2 tensor and make it a rank-3 tensor by splitting its first dimension. This function is used for
    extracting back the physical dimension of a reshaped tensor.
    :param tensor: rank-2 tensor
    :param physicalDimension: the physical dimension of the tensor.
    :return: rank-3 new tensor such that:
             newTensor.shape = (oldTensor.shape[0], oldTensor.shape[0] / physicalDimension, oldTensor.shape[1])
    """
    if len(tensor.shape) != 2:
        raise IndexError('Error: 00003')
    newTensor = np.reshape(tensor, [physicalDimension, int(tensor.shape[0] / physicalDimension), tensor.shape[1]])
    return newTensor


def rank_3_rank_n(tensor, oldShape):
    """
    Returning a tensor to its original rank-N rank.
    :param tensor: rank-3 tensor
    :param oldShape: the tensor's original shape
    :return: the tensor in its original shape.
    """
    newTensor = np.reshape(tensor, oldShape)
    return newTensor


def truncation_svd(tensor, leftIdx, rightIdx, keepS=None, maxEigenvalNumber=None):
    """
    Taking a rank-N tensor reshaping it to rank-2 tensor and preforming an SVD operation with/without truncation.
    :param tensor: the tensor
    :param leftIdx: indices to move into 0th index
    :param rightIdx: indices to move into 1st index
    :param keepS: if not None: will return U, S, V^(dagger). if None: will return U * sqrt(S), sqrt(S) * V^(dagger)
    :param maxEigenvalNumber: maximal number of eigenvalues to keep (truncation)
    :return: U, S, V^(dagger) or U * sqrt(S), sqrt(S) * V^(dagger)
    """
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
    """
    Applying Imaginary Time Evolution (ITE) on a pair of interacting tensors and returning a rank-4 tensor \theta with
    physical bond dimensions d(i') and d(j') and shape (Q1, d(i'), d(j'), Q2). Q1, Q2 are the dimensions of the QR and
    LQ matrices. The shape of the unitaryGate should be (d(i), d(j), d(i'), d(j')).
    :param iTensor: the left tensor
    :param jTensor: the right tensor
    :param middleWeightVector: the lambda weight associated with the left and right tensors common edge
    :param commonEdge: the tensors common edge
    :param timeStep: the ITE time step
    :param interactionConst: list of interaction constants J_{ij} (len(List) = # of edges)
    :param fieldConst: the field constant usually written as h
    :param iOperators: the operators associated with the i^th tensor in the Hamiltonian
    :param jOperators: the operators associated with the j^th tensor in the Hamiltonian
    :param fieldOperators: the operators associated with the field term in the Hamiltonian
    :return: A rank-4 tensor with shape (Q1, d(i'), d(j'), Q2)
    """
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
    """
    :param tensor: the tensor
    :return: the norm
    """
    tensorConj = np.conj(cp.copy(tensor))
    idx = list(range(len(tensor.shape)))
    norm = np.sqrt(np.einsum(tensor, idx, tensorConj, idx))
    return norm
