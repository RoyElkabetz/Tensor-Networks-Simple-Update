import numpy as np
import copy as cp
from scipy import linalg


########################################################################################################################
#                                                                                                                      #
#                                              SIMPLE UPDATE ALGORITHM                                                 #
#                                                                                                                      #
########################################################################################################################


def simpleUpdate(tensors,
                 weights,
                 timeStep,
                 interactionConst,
                 fieldConst,
                 iOperators,
                 jOperators,
                 fieldOperators,
                 smat,
                 Dmax,
                 type,
                 graph=None,
                 singleEdge=None):
    """
    The Simple Update algorithm implementation on a general finite tensor network specified by a structure matrix
    :param tensors: list of tensors in the tensor network [T1, T2, T3, ..., Tn]
    :param weights: list of lambda weights [L1, L2, ..., Lm]
    :param timeStep: Imaginary Time Evolution (ITE) time step
    :param interactionConst: J_{ij} constants in the Hamiltonian
    :param fieldConst: field constant in the Hamiltonian
    :param iOperators: the operators associated with the i^th tensor in the Hamiltonian
    :param jOperators: the operators associated with the j^th tensor in the Hamiltonian
    :param fieldOperators: the operators associated with the field term in the Hamiltonian
    :param smat: tensor network structure matrix
    :param Dmax: maximal bond dimension
    :param type: type of algorithm to use 'BP' or 'SU'
    :param graph: the tensor network dual double-edge factor graph
    :param singleEdge: run a single su step over that specific edge
    :return: updated tensors list and weights list
    """
    tensors = cp.deepcopy(tensors)
    weights = cp.deepcopy(weights)
    n, m = np.shape(smat)


    for Ek in range(m):
        lambda_k = weights[Ek]

        # Find tensors Ti, Tj and their corresponding indices connected along edge Ek.
        Ti, Tj = getTensors(Ek, tensors, smat)

        # collect edges and remove the Ek edge from both lists
        iEdgesNidx, jEdgesNidx = getTensorsEdges(Ek, smat)

        # absorb environment (lambda weights) into tensors
        Ti[0] = absorbWeights(Ti[0], iEdgesNidx, weights)
        Tj[0] = absorbWeights(Tj[0], jEdgesNidx, weights)

        # permuting the indices associated with edge Ek tensors Ti, Tj with their 1st index
        Ti = indexPermute(Ti)
        Tj = indexPermute(Tj)

        # Group all virtual indices Em!=Ek to form Pl, Pr MPS tensors
        Pl = rankNrank3(Ti[0])
        Pr = rankNrank3(Tj[0])

        # SVD decomposing of Pl, Pr to obtain Q1, R and Q2, L sub-tensors, respectively
        R, sr, Q1 = truncationSVD(Pl, [0, 1], [2], keepS='yes')
        L, sl, Q2 = truncationSVD(Pr, [0, 1], [2], keepS='yes')
        R = R.dot(np.diag(sr))
        L = L.dot(np.diag(sl))

        # RQ decomposition of Pl, Pr to obtain R, Q1 and L, Q2 sub-tensors, respectively (needs fixing)
        #R, Q1 = linalg.rq(np.reshape(Pl, [Pl.shape[0] * Pl.shape[1], Pl.shape[2]]))
        #L, Q2 = linalg.rq(np.reshape(Pr, [Pr.shape[0] * Pr.shape[1], Pr.shape[2]]))

        # reshaping R and L into rank 3 tensors with shape (physical_dim, Ek_dim, Q(1/2).shape[0])
        i_physical_dim = Ti[0].shape[0]
        j_physical_dim = Tj[0].shape[0]
        R = rank2rank3(R, i_physical_dim)  # (i, Ek, Q1) (following the dimensions)
        L = rank2rank3(L, j_physical_dim)  # (j, Ek, Q2)

        # Contract the ITE gate with R, L, and lambda_k to form theta tensor.
        theta = imaginaryTimeEvolution(R,
                                       L,
                                       lambda_k,
                                       Ek,
                                       timeStep,
                                       interactionConst,
                                       fieldConst,
                                       iOperators,
                                       jOperators,
                                       fieldOperators)  # (Q1, i', j', Q2)

        # Obtain R', L', lambda'_k tensors by applying an SVD to theta
        if type == 'SU':
            R_tild, lambda_k_tild, L_tild = truncationSVD(theta, [0, 1], [2, 3], keepS='yes', maxEigenvalNumber=Dmax) # with truncation
        if type == 'BP':
            R_tild, lambda_k_tild, L_tild = truncationSVD(theta, [0, 1], [2, 3], keepS='yes') # without truncation
        # (Q1 * i', D') # (D', D') # (D', j' * Q2)

        # reshaping R_tild and L_tild back to rank 3 tensor
        R_tild = np.reshape(R_tild, (Q1.shape[0], i_physical_dim, R_tild.shape[1]))  # (Q1, i', D')
        R_tild = np.transpose(R_tild, [1, 2, 0])  # (i', D', Q1)
        L_tild = np.reshape(L_tild, (L_tild.shape[0], j_physical_dim, Q2.shape[0]))  # (D', j', Q2)
        L_tild = np.transpose(L_tild, [1, 0, 2])  # (j', D', Q2)

        # Glue back the R', L', sub-tensors to Q1, Q2, respectively, to form updated tensors P'l, P'r.
        Pl_prime = np.einsum('ijk,kl->ijl', R_tild, Q1)
        Pr_prime = np.einsum('ijk,kl->ijl', L_tild, Q2)

        # Reshape back the P`l, P`r to the original rank-(z + 1) tensors Ti, Tj
        Ti_new_shape = list(Ti[0].shape)
        Ti_new_shape[1] = len(lambda_k_tild)
        Tj_new_shape = list(Tj[0].shape)
        Tj_new_shape[1] = len(lambda_k_tild)
        Ti[0] = rank3rankN(Pl_prime, Ti_new_shape)
        Tj[0] = rank3rankN(Pr_prime, Tj_new_shape)

        # permuting back the legs of Ti and Tj
        Ti = indexPermute(Ti)
        Tj = indexPermute(Tj)

        # Remove bond matrices lambda_m from virtual legs m != Ek to obtain the updated tensors Ti~, Tj~.
        Ti[0] = absorbInverseWeights(Ti[0], iEdgesNidx, weights)
        Tj[0] = absorbInverseWeights(Tj[0], jEdgesNidx, weights)

        # Normalize and save new Ti Tj and lambda_k
        tensors[Ti[1][0]] = Ti[0] / tensorNorm(Ti[0])
        tensors[Tj[1][0]] = Tj[0] / tensorNorm(Tj[0])
        weights[Ek] = lambda_k_tild / np.sum(lambda_k_tild)


    return tensors, weights


########################################################################################################################
#                                                                                                                      #
#                                        SIMPLE UPDATE AUXILIARY FUNCTIONS                                             #
#                                                                                                                      #
########################################################################################################################


def getTensors(edge, tensors, smat):
    """
    Given an edge collect neighboring tensors and returns their copies.
    :param edge: edge number {0, 1, ..., m-1}.
    :param tensors: list of tensors.
    :param smat: structure matrix (n x m).
    :return: two lists of Ti Tj tensors, [tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    tensorNumber = np.nonzero(smat[:, edge])[0]
    tensorIndexAlongEdge = smat[tensorNumber, edge]
    Ti = [cp.copy(tensors[tensorNumber[0]]),
          [tensorNumber[0], 'tensor_number'],
          [tensorIndexAlongEdge[0], 'tensor_index_along_edge']
          ]
    Tj = [cp.copy(tensors[tensorNumber[1]]),
          [tensorNumber[1], 'tensor_number'],
          [tensorIndexAlongEdge[1], 'tensor_index_along_edge']
          ]
    return Ti, Tj


def getConjTensors(edge, tensors, smat):
    """
    Given an edge collect neighboring tensors.
    :param edge: edge number {0, 1, ..., m-1}.
    :param tensors: list of tensors.
    :param smat: structure matrix (n x m).
    :return: two lists of Ti Tj conjugate tensors, [tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    tensorNumber = np.nonzero(smat[:, edge])[0]
    tensorIndexAlongEdge = smat[tensorNumber, edge]
    Ti = [np.conj(cp.copy(tensors[tensorNumber[0]])),
          [tensorNumber[0], 'tensor_number'],
          [tensorIndexAlongEdge[0], 'tensor_index_along_edge']
          ]
    Tj = [np.conj(cp.copy(tensors[tensorNumber[1]])),
          [tensorNumber[1], 'tensor_number'],
          [tensorIndexAlongEdge[1], 'tensor_index_along_edge']
          ]
    return Ti, Tj


def getTensorsEdges(edge, smat):
    """
    Given an edge, collect neighboring tensors edges and indices
    :param edge: edge number {0, 1, ..., m-1}.
    :param smat: structure matrix (n x m).
    :return: two lists of Ti, Tj edges and associated indices with 'edge' and its index removed.
    """
    tensorNumber = np.nonzero(smat[:, edge])[0]
    iEdgesNidx = [list(np.nonzero(smat[tensorNumber[0], :])[0]),
                  list(smat[tensorNumber[0], np.nonzero(smat[tensorNumber[0], :])[0]])
                  ]  # [edges, indices]
    jEdgesNidx = [list(np.nonzero(smat[tensorNumber[1], :])[0]),
                  list(smat[tensorNumber[1], np.nonzero(smat[tensorNumber[1], :])[0]])
                  ]  # [edges, indices]
    # remove 'edge' and its associated index from both i, j lists.
    iEdgesNidx[0].remove(edge)
    iEdgesNidx[1].remove(smat[tensorNumber[0], edge])
    jEdgesNidx[0].remove(edge)
    jEdgesNidx[1].remove(smat[tensorNumber[1], edge])
    return iEdgesNidx, jEdgesNidx


def getEdges(tensorIndex, smat):
    """
    Given an index of a tensor, return all of its edges and associated indices.
    :param tensorIndex: the tensor index in the structure matrix
    :param smat: structure matrix
    :return: list of two lists [[edges], [indices]].
    """
    edges = np.nonzero(smat[tensorIndex, :])[0]
    indices = smat[tensorIndex, edges]
    return [edges, indices]


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


def absorbWeights(tensor, edgesNidx, weights):
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


def absorbWeightsTwoSiteExpectationWithRectangularEnvironment(tensor, edgesNidx, weights, edgesINenv, edgesOUTenv):
    """
    Given a tensor and two lists of edges inside and on the boundary (outside) of rectangular environment
    of two site expectation, this auxilary function absorb the tensor neighboring weights according to edges environment
    lists. If edge is inside the rectangular environment, then its 'sqrt(lambda weight)' is absorbed. If edge is
    on the boundary (outside) of the rectangular environment, then its 'lambda weight' is absorbed.
    :param tensor: tensor inside rectangular environment
    :param edgesNidx: list of two lists [[edges], [indices]]
    :param weights: list of lambda weights
    :param edgesINenv: list of edges inside the rectangular environment
    :param edgesOUTenv: list of edges on the boundary of the rectangular environment
    :return: the new tensor list [new_tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    for i in range(len(edgesNidx[0])):
        if edgesNidx[0][i] in edgesINenv:
            tensor = np.einsum(tensor, list(range(len(tensor.shape))), np.sqrt(weights[int(edgesNidx[0][i])]),
                               [int(edgesNidx[1][i])], list(range(len(tensor.shape))))
        elif edgesNidx[0][i] in edgesOUTenv:
            tensor = np.einsum(tensor, list(range(len(tensor.shape))), weights[int(edgesNidx[0][i])],
                               [int(edgesNidx[1][i])], list(range(len(tensor.shape))))
        else:
            raise IndexError('Error: 00001')
    return tensor


def absorbInverseWeights(tensor, edgesNidx, weights):
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


def indexPermute(tensor):
    """
    Swapping the 'tensor_index_along_edge' index with the 1st index
    :param tensor: [tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    :return: the list with the permuted tensor [permuted_tensor, [#, 'tensor_number'], [#, 'tensor_index_along_edge']]
    """
    permutation = np.array(list(range(len(tensor[0].shape))))
    permutation[[1, tensor[2][0]]] = permutation[[tensor[2][0], 1]]
    tensor[0] = np.transpose(tensor[0], permutation)
    return tensor


def rankNrank3(tensor):
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


def rank2rank3(tensor, physicalDimension):
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


def rank3rankN(tensor, oldShape):
    """
    Returning a tensor to its original rank-N rank.
    :param tensor: rank-3 tensor
    :param oldShape: the tensor's original shape
    :return: the tensor in its original shape.
    """
    newTensor = np.reshape(tensor, oldShape)
    return newTensor


def truncationSVD(tensor, leftIdx, rightIdx, keepS=None, maxEigenvalNumber=None):
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


def imaginaryTimeEvolution(iTensor,
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


def tensorNorm(tensor):
    """
    :param tensor: the tensor
    :return: the norm
    """
    tensorConj = np.conj(cp.copy(tensor))
    idx = list(range(len(tensor.shape)))
    norm = np.sqrt(np.einsum(tensor, idx, tensorConj, idx))
    return norm


def updateDEFG(edge, tensors, weights, smat, doubleEdgeFactorGraph):
    """
    DEFG update (description needs to be added)
    :param edge:
    :param tensors:
    :param weights:
    :param smat:
    :param doubleEdgeFactorGraph:
    :return: None
    """
    iFactor, jFactor = getTensors(edge, tensors, smat)
    iEdges, jEdges = getAllTensorsEdges(edge, smat)
    iFactor[0] = absorbSqrtWeights(cp.deepcopy(iFactor[0]), iEdges, weights)
    jFactor[0] = absorbSqrtWeights(cp.deepcopy(jFactor[0]), jEdges, weights)
    doubleEdgeFactorGraph.factors['f' + str(iFactor[1][0])][1] = iFactor[0]
    doubleEdgeFactorGraph.factors['f' + str(jFactor[1][0])][1] = jFactor[0]
    doubleEdgeFactorGraph.nodes['n' + str(edge)][0] = len(weights[edge])


