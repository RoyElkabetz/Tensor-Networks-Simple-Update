
import numpy as np
import ncon as ncon
import copy as cp
from scipy import linalg
import StructureMatrixGenerator as tnf
import ncon_lists_generator as nlg


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

    if singleEdge:
        Ek = singleEdge
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
        # R, Q1 = linalg.rq(np.reshape(Pl, [Pl.shape[0] * Pl.shape[1], Pl.shape[2]]))
        # L, Q2 = linalg.rq(np.reshape(Pr, [Pr.shape[0] * Pr.shape[1], Pr.shape[2]]))

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
            R_tild, lambda_k_tild, L_tild = truncationSVD(theta, [0, 1], [2, 3], keepS='yes',
                                                          maxEigenvalNumber=Dmax)  # with truncation
        if type == 'BP':
            R_tild, lambda_k_tild, L_tild = truncationSVD(theta, [0, 1], [2, 3], keepS='yes')  # without truncation
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

        if type == 'BP':
            tensors, weights = singleEdgeBPU(tensors, weights, smat, Dmax, Ek, graph)



    else:
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

            # single edge BP update (uncomment for single edge BP implemintation)
            if type == 'BP':
                tensors, weights = singleEdgeBPU(tensors, weights, smat, Dmax, Ek, graph)

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


########################################################################################################################
#                                                                                                                      #
#                                        SIMPLE UPDATE EXPECTATIONS & RDMs                                             #
#                                                                                                                      #
########################################################################################################################


def singleSiteExpectation(tensorIndex, tensors, weights, smat, localOp):
    """
    This function calculates the local expectation value of a single tensor network site using the weights as
    environment.
    :param tensorIndex: the index of the tensor in the structure matrix
    :param tensors: list of tensors in the tensorNet
    :param weights: list of weights
    :param smat: the structure matrix
    :param localOp: the local operator for the expectation value
    :return: single site expectation
    """
    edgeNidx = getEdges(tensorIndex, smat)
    site = absorbWeights(cp.copy(tensors[tensorIndex]), edgeNidx, weights)
    siteConj = absorbWeights(np.conj(cp.copy(tensors[tensorIndex])), edgeNidx, weights)
    normalization = siteNorm(tensorIndex, tensors, weights, smat)

    # setting lists for ncon.ncon
    siteIdx = list(range(len(site.shape)))
    siteConjIdx = list(range(len(siteConj.shape)))
    siteConjIdx[0] = len(siteConj.shape)
    localOpIdx = [siteConjIdx[0], siteIdx[0]]
    expectation = ncon.ncon([site, siteConj, localOp], [siteIdx, siteConjIdx, localOpIdx]) / normalization
    return expectation


def siteNorm(tensorIndex, tensors, weights, smat):
    """
    Calculate the normalization of a single tensor network site using the weights as environment (sam as calculating
    this site expectation with np.eye(d)).
    :param tensorIndex: the index of the tensor in the structure matrix
    :param tensors: list of tensors in the tensorNet
    :param weights: list of weights
    :param smat: the structure matrix
    :return: site normalization
    """
    edgeNidx = getEdges(tensorIndex, smat)
    site = absorbWeights(cp.copy(tensors[tensorIndex]), edgeNidx, weights)
    siteConj = absorbWeights(np.conj(cp.copy(tensors[tensorIndex])), edgeNidx, weights)
    normalization = np.einsum(site, list(range(len(site.shape))), siteConj, list(range(len(siteConj.shape))))
    return normalization


def doubleSiteExpectation(commonEdge, tensors, weights, smat, LocalOp):
    """
    Calculating the normalized double site expectation value <psi|O|psi> / <psi|psi> on a given common edge of two
    tensor network sites. The environment of the two sites are calculated using the simple update weights.
    :param commonEdge: the two sites common edge
    :param tensors: the TensorNet list of tensors
    :param weights: the TensorNet list of weights
    :param smat: structure matrix
    :param LocalOp: the local operator
    :return: double site expectation
    """
    commonWeights = cp.copy(weights[commonEdge])
    siteI, siteJ = getTensors(commonEdge, tensors, smat)
    siteIconj, siteJconj = getConjTensors(commonEdge, tensors, smat)
    edgeNidxI, edgeNidxJ = getTensorsEdges(commonEdge, smat)
    siteI[0] = absorbWeights(siteI[0], edgeNidxI, weights)
    siteJ[0] = absorbWeights(siteJ[0], edgeNidxJ, weights)
    siteIconj[0] = absorbWeights(siteIconj[0], edgeNidxI, weights)
    siteJconj[0] = absorbWeights(siteJconj[0], edgeNidxJ, weights)

    # setting lists for ncon.ncon
    s = 10000
    t = 20000
    commonWeightIdx = [t, t + 1]
    commonWeightConjIdx = [t + 2, t + 3]
    LocalOpIdx = [s, s + 1, s + 2, s + 3]  # (i, j, i', j')

    siteIidx = list(range(len(siteI[0].shape)))
    siteIconjIdx = list(range(len(siteIconj[0].shape)))
    siteIidx[0] = LocalOpIdx[0]  # i
    siteIconjIdx[0] = LocalOpIdx[2]  # i'
    siteIidx[siteI[2][0]] = commonWeightIdx[0]
    siteIconjIdx[siteIconj[2][0]] = commonWeightConjIdx[0]

    siteJidx = list(range(len(siteI[0].shape) + 1, len(siteI[0].shape) + 1 + len(siteJ[0].shape)))
    siteJconjIdx = list(range(len(siteIconj[0].shape) + 1, len(siteIconj[0].shape) + 1 + len(siteJconj[0].shape)))
    siteJidx[0] = LocalOpIdx[1]  # j
    siteJconjIdx[0] = LocalOpIdx[3]  # j'
    siteJidx[siteJ[2][0]] = commonWeightIdx[1]
    siteJconjIdx[siteJconj[2][0]] = commonWeightConjIdx[1]

    # double site expectation calculation
    tensorsList = [siteI[0], siteIconj[0], siteJ[0], siteJconj[0], LocalOp, np.diag(commonWeights), np.diag(commonWeights)]
    indicesList = [siteIidx, siteIconjIdx, siteJidx, siteJconjIdx, LocalOpIdx, commonWeightIdx, commonWeightConjIdx]
    norm = doubleSiteNorm(commonEdge, tensors, weights, smat)
    expectation = ncon.ncon(tensorsList, indicesList) / norm
    return expectation


def doubleSiteNorm(commonEdge, tensors, weights, smat):
    """
    Calculating the double site normalization <psi|psi> of two TensorNet sites sharing a common edge using the simple update
    weights as environment.
    :param commonEdge: the two sites common edge
    :param tensors: the TensorNet list of tensors
    :param weights: the TensorNet list of weights
    :param smat: structure matrix
    :return: double site normalization
    """
    commonWeights = cp.copy(weights[commonEdge])
    siteI, siteJ = getTensors(commonEdge, tensors, smat)
    siteIconj, siteJconj = getConjTensors(commonEdge, tensors, smat)
    edgeNidxI, edgeNidxJ = getTensorsEdges(commonEdge, smat)
    siteI[0] = absorbWeights(siteI[0], edgeNidxI, weights)
    siteJ[0] = absorbWeights(siteJ[0], edgeNidxJ, weights)
    siteIconj[0] = absorbWeights(siteIconj[0], edgeNidxI, weights)
    siteJconj[0] = absorbWeights(siteJconj[0], edgeNidxJ, weights)

    # setting lists for ncon.ncon
    s = 10000
    t = 20000
    commonWeightIdx = [t, t + 1]
    commonWeightConjIdx = [t + 2, t + 3]

    siteIidx = list(range(len(siteI[0].shape)))
    siteIconjIdx = list(range(len(siteIconj[0].shape)))
    siteIidx[0] = s  # i
    siteIconjIdx[0] = s  # i'
    siteIidx[siteI[2][0]] = commonWeightIdx[0]
    siteIconjIdx[siteIconj[2][0]] = commonWeightConjIdx[0]

    siteJidx = list(range(len(siteI[0].shape) + 1, len(siteI[0].shape) + 1 + len(siteJ[0].shape)))
    siteJconjIdx = list(range(len(siteIconj[0].shape) + 1, len(siteIconj[0].shape) + 1 + len(siteJconj[0].shape)))
    siteJidx[0] = s + 1  # j
    siteJconjIdx[0] = s + 1  # j'
    siteJidx[siteJ[2][0]] = commonWeightIdx[1]
    siteJconjIdx[siteJconj[2][0]] = commonWeightConjIdx[1]

    # double site normalization calculation
    tensorsList = [siteI[0], siteIconj[0], siteJ[0], siteJconj[0], np.diag(commonWeights), np.diag(commonWeights)]
    indicesList = [siteIidx, siteIconjIdx, siteJidx, siteJconjIdx, commonWeightIdx, commonWeightConjIdx]
    norm = ncon.ncon(tensorsList, indicesList)
    return norm


def doubleSiteRDM(commonEdge, tensors, weights, smat):
    """
    Calculating the double site reduced density matrix rho_{ii', jj'} using the simple update weights as environments.
    :param commonEdge: the two tensorNet
    :param tensors: the TensorNet tensors list
    :param weights: the TensorNet weights list
    :param smat: structure matrix
    :return: two site RDM rho_{i * i', j * j'} when {i, j} relate to the ket and {i', j'} to the bra.
    """
    commonWeight = cp.copy(weights[commonEdge])
    SiteI, SiteJ = getTensors(commonEdge, tensors, smat)
    SiteIconj, siteJconj = getConjTensors(commonEdge, tensors, smat)
    edgeNidxI, edgeNidxJ = getTensorsEdges(commonEdge, smat)
    SiteI[0] = absorbWeights(SiteI[0], edgeNidxI, weights)
    SiteJ[0] = absorbWeights(SiteJ[0], edgeNidxJ, weights)
    SiteIconj[0] = absorbWeights(SiteIconj[0], edgeNidxI, weights)
    siteJconj[0] = absorbWeights(siteJconj[0], edgeNidxJ, weights)

    ## setting lists of tensors and indices for ncon.ncon
    t = 20000
    commonEdgeIdx = [t, t + 1]
    commonEdgeConjIdx = [t + 2, t + 3]

    siteIidx = list(range(len(SiteI[0].shape)))
    siteIconjIdx = list(range(len(SiteIconj[0].shape)))
    siteIidx[0] = -1  # i
    siteIconjIdx[0] = -2  # i'
    siteIidx[SiteI[2][0]] = commonEdgeIdx[0]
    siteIconjIdx[SiteIconj[2][0]] = commonEdgeConjIdx[0]

    siteJidx = list(range(len(SiteI[0].shape) + 1, len(SiteI[0].shape) + 1 + len(SiteJ[0].shape)))
    siteJconjIdx = list(range(len(SiteIconj[0].shape) + 1, len(SiteIconj[0].shape) + 1 + len(siteJconj[0].shape)))
    siteJidx[0] = -3  # j
    siteJconjIdx[0] = -4  # j'
    siteJidx[SiteJ[2][0]] = commonEdgeIdx[1]
    siteJconjIdx[siteJconj[2][0]] = commonEdgeConjIdx[1]

    # two site expectation calculation
    tensors = [SiteI[0], SiteIconj[0], SiteJ[0], siteJconj[0], np.diag(commonWeight), np.diag(commonWeight)]
    indices = [siteIidx, siteIconjIdx, siteJidx, siteJconjIdx, commonEdgeIdx, commonEdgeConjIdx]
    rdm = ncon.ncon(tensors, indices)  # rho_{i, i', j, j'}
    rdm = rdm.reshape(rdm.shape[0] * rdm.shape[1], rdm.shape[2] * rdm.shape[3])  # rho_{i * i', j * j'}
    rdm /= np.trace(rdm)
    return rdm


def PEPSdoubleSiteExpectationRectEnvironment(commonEdge, envSize, networkShape, tensors, weights, smat, localOp):
    TT = cp.deepcopy(tensors)
    TTconj = conjTensorNet(cp.deepcopy(tensors))
    LL = cp.deepcopy(weights)
    p = localOp.shape[0]
    Iop = np.eye(p ** 2).reshape(p, p, p, p)

    # get th environment matrix and the lists of inside and outside edges
    emat = tnf.PEPS_OBC_edge_rect_env(commonEdge, smat, networkShape, envSize)
    inside, outside = tnf.PEPS_OBC_divide_edge_regions(emat, smat)
    omat = np.arange(smat.shape[0]).reshape(emat.shape)
    tensors_indices = omat[np.nonzero(emat > -1)]

    # absorb edges
    for t in tensors_indices:
        edge_leg = getEdges(t, smat)
        TT[t] = absorbWeightsTwoSiteExpectationWithRectangularEnvironment(TT[t], edge_leg, LL, inside, outside)
        TTconj[t] = absorbWeightsTwoSiteExpectationWithRectangularEnvironment(TTconj[t], edge_leg, LL, inside, outside)

    # lists and ncon
    t_list, i_list, o_list = nlg.ncon_list_generator_two_site_expectation_with_env_peps_obc(TT, TTconj, localOp, smat, emat, commonEdge, tensors_indices, inside, outside)
    t_list_n, i_list_n, o_list_n = nlg.ncon_list_generator_two_site_expectation_with_env_peps_obc(TT, TTconj, Iop, smat, emat, commonEdge, tensors_indices, inside, outside)
    expec = ncon.ncon(t_list, i_list, o_list)
    norm = ncon.ncon(t_list_n, i_list_n, o_list_n)
    expectation = expec / norm
    return expectation


def PEPSdoubleSiteExactExpectation(tensors, weights, smat, commonEdge, localOp):
    """
    Caluclating PEPS local operator exact expectation value by contracting the whole TensorNet.
    :param tensors: the TensorNet tensors list
    :param weights: the TensorNet weights list
    :param smat: structure matrix
    :param commonEdge: the common edge of the tow tensors
    :param localOp: the local operator
    :return: exact expectation value
    """
    tensors = cp.deepcopy(tensors)
    weights = cp.deepcopy(weights)
    tensorsConj = conjTensorNet(tensors)
    tensorsA = absorbAllTensorNetWeights(tensors, weights, smat)
    tensorsConjA = absorbAllTensorNetWeights(tensorsConj, weights, smat)
    tensorsList, idxList = nlg.ncon_list_generator_two_site_exact_expectation_peps(tensorsA, tensorsConjA, smat, commonEdge, localOp)
    tensorsListNorm, idxListNorm = nlg.ncon_list_generator_braket_peps(tensorsA, tensorsConjA, smat)
    exactExpectation = ncon.ncon(tensorsList, idxList) / ncon.ncon(tensorsListNorm, idxListNorm)
    return exactExpectation


def conjTensorNet(tensors):
    """
    Given a TensorNet list of tensors returns the list of complex conjugate tensors
    :param tensors: the TensorNet list of tensors
    :return: list of complex conjugate tensors
    """
    tensorsConj = []
    for i in range(len(tensors)):
        tensorsConj.append(np.conj(tensors[i]))
    return tensorsConj


def energyPerSite(tensors, weights, smat, interactionConst, filedConst, iSiteOp, jSiteOp, fieldOp):
    """
    Calculating the energy per site of a given TensorNet in the simple update method with weights as environments.
    :param tensors: list of tensors in the TensorNet
    :param weights: list of weights in the TensorNet
    :param smat: structure matrix
    :param interactionConst: the J_{ij} interaction constants of the Hamiltonian
    :param filedConst: the field constant h
    :param iSiteOp: i site operators i.e. [X, Y, Z]
    :param jSiteOp: j site operators i.e. [X, Y, Z]
    :param fieldOp: field operators i.e. [X]
    :return: the energy per site of a TensorNet
    """
    energy = 0
    d = iSiteOp[0].shape[0]
    Aij = np.zeros((d ** 2, d ** 2), dtype=complex)
    for i in range(len(iSiteOp)):
        Aij += np.kron(iSiteOp[i], jSiteOp[i])
    n, m = np.shape(smat)
    for edge in range(m):
        ijLocalOp = np.reshape(-interactionConst[edge] * Aij - 0.25 * filedConst
                         * (np.kron(np.eye(d), fieldOp) + np.kron(fieldOp, np.eye(d))), (d, d, d, d))
        energy += doubleSiteExpectation(edge, tensors, weights, smat, ijLocalOp)
    energy /= n
    return energy


def PEPSenergyPerSiteWithRectEnvironment(networkShape, envSize, tensors, weights, smat, Jk, h, iOp, jOp, fieldOp):
    """

    :param networkShape:
    :param envSize:
    :param tensors:
    :param weights:
    :param smat:
    :param Jk:
    :param h:
    :param iOp:
    :param jOp:
    :param fieldOp:
    :return:
    """
    tensors = cp.deepcopy(tensors)
    weights = cp.deepcopy(weights)
    energy = 0
    d = iOp[0].shape[0]
    Aij = np.zeros((d ** 2, d ** 2), dtype=complex)
    for i in range(len(iOp)):
        Aij += np.kron(iOp[i], jOp[i])
    n, m = np.shape(smat)
    for Ek in range(m):
        print(Ek)
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(d), fieldOp) + np.kron(fieldOp, np.eye(d))), (d, d, d, d))
        energy += PEPSdoubleSiteExpectationRectEnvironment(Ek, envSize, networkShape, tensors, weights, smat, Oij)
    energy /= n
    return energy


def PEPSexactEnergyPerSite(tensors, weights, smat, Jk, h, iOp, jOp, fieldOp):
    """

    :param tensors:
    :param weights:
    :param smat:
    :param Jk:
    :param h:
    :param iOp:
    :param jOp:
    :param fieldOp:
    :return:
    """
    energy = 0
    d = iOp[0].shape[0]
    Aij = np.zeros((d ** 2, d ** 2), dtype=complex)
    for i in range(len(iOp)):
        Aij += np.kron(iOp[i], jOp[i])
    n, m = np.shape(smat)
    for Ek in range(m):
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(d), fieldOp) + np.kron(fieldOp, np.eye(d))), (d, d, d, d))
        energy += PEPSdoubleSiteExactExpectation(tensors, weights, smat, Ek, Oij)
    energy /= n
    return energy


def BPenergyPerSite(defg, smat, Jk, h, iOp, jOp, fieldOp):
    """
    Calculating a TensorNet energy per site using the DEFG and its factor beliefs
    :param defg: the TensorNet dual double-edge factor graph
    :param smat: structure matrix
    :param Jk: Hamiltonian's interaction constants J_{ij}
    :param h: Hamiltonian's  field constant
    :param iOp: Hamiltonian's i^th tensor operators
    :param jOp: Hamiltonian's j^th tensor operators
    :param fieldOp:  Hamiltonian's field operators
    :return: energy per site
    """
    energy = 0
    d = iOp[0].shape[0]
    Aij = np.zeros((d ** 2, d ** 2), dtype=complex)
    for i in range(len(iOp)):
        Aij += np.kron(iOp[i], jOp[i])
    n, m = np.shape(smat)
    for commonEdge in range(m):
        Oij = np.reshape(-Jk[commonEdge] * Aij - 0.25 * h * (np.kron(np.eye(d), fieldOp) + np.kron(fieldOp, np.eye(d))),
                         (d, d, d, d))
        tensors = np.nonzero(smat[:, commonEdge])[0]
        fi_belief, fj_belief = defg.twoFactorsBelief('f' + str(tensors[0]), 'f' + str(tensors[1]))
        fi_idx = list(range(len(fi_belief.shape)))
        fj_idx = list(range(len(fi_belief.shape), len(fi_belief.shape) + len(fj_belief.shape)))
        Oij_idx = [10000, 10001, 10002, 10003]  # Oij_{i, j, i', j'}
        fi_idx[0] = Oij_idx[0]
        fi_idx[1] = Oij_idx[2]
        fj_idx[0] = Oij_idx[1]
        fj_idx[1] = Oij_idx[3]
        iEdgeNidx, jEdgeNidx = getTensorsEdges(commonEdge, smat)
        for leg_idx, leg in enumerate(iEdgeNidx[1]):
            fi_idx[2 * leg + 1] = fi_idx[2 * leg]
        for leg_idx, leg in enumerate(jEdgeNidx[1]):
            fj_idx[2 * leg + 1] = fj_idx[2 * leg]
        edge_legs = smat[np.nonzero(smat[:, commonEdge])[0], commonEdge]
        fi_idx[2 * edge_legs[0]] = fj_idx[2 * edge_legs[1]]
        fi_idx[2 * edge_legs[0] + 1] = fj_idx[2 * edge_legs[1] + 1]
        siteEnergy = ncon.ncon([fi_belief, fj_belief, Oij], [fi_idx, fj_idx, Oij_idx])
        norm = ncon.ncon([fi_belief, fj_belief, np.eye(d ** 2).reshape((d, d, d, d))], [fi_idx, fj_idx, Oij_idx])
        siteNormelizedEnergy = siteEnergy / norm
        energy += siteNormelizedEnergy
    energy /= n
    return energy


def BPdoubleSiteRDM(commonEdge, graph, smat):
    """
    Given two tensors common edge in a TensorNet and its dual DEFG this function returns the reduced density matrix
    rho_{i * j, i' * j'} where i,j relate to the ket and i',j' relate to the bra.
    :param commonEdge: the two tensors common edge
    :param graph: the TensorNet dual DEFG
    :param smat: structure matrix
    :return: rdm as in rho_{i * j, i' * j'}
    """
    tensors = np.nonzero(smat[:, commonEdge])[0]
    fi_belief, fj_belief = graph.twoFactorsBelief('f' + str(tensors[0]), 'f' + str(tensors[1]))
    fi_idx = list(range(len(fi_belief.shape)))
    fj_idx = list(range(len(fi_belief.shape), len(fi_belief.shape) + len(fj_belief.shape)))
    fi_idx[0] = -1  # i
    fi_idx[1] = -3  # i'
    fj_idx[0] = -2  # j
    fj_idx[1] = -4  # j'
    iEdgeNidx, jEdgeNidx = getTensorsEdges(commonEdge, smat)
    for leg_idx, leg in enumerate(iEdgeNidx[1]):
        fi_idx[2 * leg + 1] = fi_idx[2 * leg]
    for leg_idx, leg in enumerate(jEdgeNidx[1]):
        fj_idx[2 * leg + 1] = fj_idx[2 * leg]
    commonEdgeIdx = smat[np.nonzero(smat[:, commonEdge])[0], commonEdge]
    fi_idx[2 * commonEdgeIdx[0]] = fj_idx[2 * commonEdgeIdx[1]]
    fi_idx[2 * commonEdgeIdx[0] + 1] = fj_idx[2 * commonEdgeIdx[1] + 1]
    rdm = ncon.ncon([fi_belief, fj_belief], [fi_idx, fj_idx])  # rho_{i, j, i', j'}
    rdm = rdm.reshape(rdm.shape[0] * rdm.shape[1], rdm.shape[2] * rdm.shape[3])  # rho_{i * j, i' * j'}
    rdm /= np.trace(rdm)  # rho_{i * j, i' * j'}
    return rdm

# fix
def BP_energy_per_site_using_factor_belief_with_environment(graph, env_size, network_shape, smat, Jk, h, Opi, Opj, Op_field):
    energy = 0
    p = Opi[0].shape[0]
    Aij = np.zeros((p ** 2, p ** 2), dtype=complex)
    Iop = np.eye(Aij.shape[0]).reshape(p, p, p, p)
    for i in range(len(Opi)):
        Aij += np.kron(Opi[i], Opj[i])
    n, m = np.shape(smat)
    for Ek in range(m):
        print('Ek = ', Ek)
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p))), (p, p, p, p))
        f_list, i_list, o_list = nlg.ncon_list_generator_two_site_expectation_with_factor_belief_env_peps_obc_efficient(Ek, graph, env_size, network_shape, smat, Oij)
        f_list_n, i_list_n, o_list_n = nlg.ncon_list_generator_two_site_expectation_with_factor_belief_env_peps_obc_efficient(Ek, graph, env_size, network_shape, smat, Iop)
        expec = ncon.ncon(f_list, i_list, o_list)
        norm = ncon.ncon(f_list_n, i_list_n, o_list_n)
        expectation = expec / norm
        energy += expectation
    energy /= n
    return energy

# if not in use, can be deleted (DEFG mean field approch to energy per site)
def BP_energy_per_site_using_rdm_belief(graph, smat, Jk, h, Opi, Opj, Op_field):
    # calculating the normalized exact energy per site(tensor)
    if graph.rdm_belief == None:
        raise IndexError('First calculate rdm beliefs')
    p = Opi[0].shape[0]
    Aij = np.zeros((p ** 2, p ** 2), dtype=complex)
    for i in range(len(Opi)):
        Aij += np.kron(Opi[i], Opj[i])
    energy = 0
    n, m = np.shape(smat)
    for Ek in range(m):
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p))), (p, p, p, p))
        tensors = np.nonzero(smat[:, Ek])[0]
        fi_belief = graph.rdm_belief[tensors[0]]
        fj_belief = graph.rdm_belief[tensors[1]]
        fij = np.einsum(fi_belief, [0, 1], fj_belief, [2, 3], [0, 2, 1, 3])
        Oij_idx = [0, 1, 2, 3]
        E = np.einsum(fij, [0, 1, 2, 3], Oij, Oij_idx)
        norm = np.einsum(fij, [0, 1, 0, 1])
        E_normalized = E / norm
        energy += E_normalized
    energy /= n
    return energy


def traceDistance(a, b):
    # returns the trace distance between the two density matrices a & b
    # d = 0.5 * norm(a - b)
    eigenvalues = np.linalg.eigvals(a - b)
    d = 0.5 * np.sum(np.abs(eigenvalues))
    return d


def singleSiteRDM(tensorIdx, tensors, weights, smat):
    """
    TensorNet single site rdm
    :param tensorIdx: the tensor index in the structure matrix
    :param tensors: the TensorNet list of tensors
    :param weights: the TensorNet list of weights
    :param smat: structure matrix
    :return: single site rdm rho_{i, i'} where i relate to the ket and i' to the bra.
    """
    edgeNidx = getEdges(tensorIdx, smat)
    tensor = absorbWeights(cp.copy(tensors[tensorIdx]), edgeNidx, weights)
    tensorConj = absorbWeights(cp.copy(np.conj(tensors[tensorIdx])),edgeNidx, weights)
    tIdx = list(range(len(tensor.shape)))
    tIdx[0] = -1
    tConjIdx = list(range(len(tensorConj.shape)))
    tConjIdx[0] = -2
    rdm = ncon.ncon([tensor, tensorConj], [tIdx, tConjIdx])
    return rdm / np.trace(rdm)


def absorbAllTensorNetWeights(tensors, weights, smat):
    n = len(tensors)
    for i in range(n):
        edgeNidx = getEdges(i, smat)
        tensors[i] = absorbSqrtWeights(tensors[i], edgeNidx, weights)
    return tensors


########################################################################################################################
#                                                                                                                      #
#                                  SIMPLE UPDATE with BELIEF PROPAGATION UPDATE (BPU)                                  #
#                                                                                                                      #
########################################################################################################################


def AllEdgesBPU(defg, tensors, weights, smat, Dmax):
    """
    Preforms the Belief Propagation Update (BPU) algorithm using the Belief Propagation Truncation (BPT) on all
    the TensorNet edges.
    :param tensors: the TensorNet list of tensors
    :param weights: the TensorNet list of weights
    :param smat: structure matrix
    :param Dmax: the maximal virtual bond dimension
    :param defg: the TensorNet dual double-edge factor graph
    :return: the updated tensors and weights lists
    """
    for edge in range(len(weights)):
        node = 'n' + str(edge)
        siteI, siteJ = getTensors(edge, tensors, smat)
        edgeNidxI, edgeNidxJ = getAllTensorsEdges(edge, smat)
        fi = absorbSqrtWeights(siteI[0], edgeNidxI, weights)
        fj = absorbSqrtWeights(siteJ[0], edgeNidxJ, weights)
        A, B = AnB_calculation(defg, fi, fj, node)
        P = find_P(A, B, Dmax)
        tensors, weights = BPtruncation(tensors, weights, P, edge, smat, Dmax)
        updateDEFG(edge, tensors, weights, smat, defg)
    return tensors, weights


def singleEdgeBPU(tensors, weights, smat, Dmax, edge, defg):
    """
    Preforms the Belief Propagation Update (BPU) algorithm using the Belief Propagation Truncation (BPT) on a
    single TensorNet edge.
    :param tensors: the TensorNet list of tensors
    :param weights: the TensorNet list of weights
    :param smat: structure matrix
    :param Dmax: the edge maximal virtual bond dimension
    :param edge: the specific edge
    :param defg: the TensorNet dual double-edge factor graph
    :return: the updated tensors and weights lists
    """
    node = 'n' + str(edge)
    siteI, siteJ = getTensors(edge, tensors, smat)
    edgeNidxI, edgeNidxJ = getAllTensorsEdges(edge, smat)
    fi = [absorbSqrtWeights(siteI[0], edgeNidxI, weights), siteI[1]]
    fj = [absorbSqrtWeights(siteJ[0], edgeNidxJ, weights), siteJ[1]]
    A, B = AnB_calculation(defg, fi, fj, node)
    P = find_P(A, B, Dmax)
    tensors, weights = BPtruncation(tensors, weights, P, edge, smat, Dmax)
    updateDEFG(edge, tensors, weights, smat, defg)
    return tensors, weights


def TNtoDEFGtransform(defg, tensors, weights, smat):
    """
    Generate the double-edge factor graph from a TensorNet
    :param defg: empty DEFG
    :param tensors: the TensorNet list of tensors
    :param weights: the TensorNet list of weights
    :param smat: structure matrix
    :return:
    """
    factorsList = absorbAllTensorNetWeights(tensors, weights, smat)
    n, m = np.shape(smat)
    for i in range(m):
        defg.add_node(len(weights[i]), 'n' + str(defg.nCounter))  # Adding virtual nodes
    for i in range(n):  # Adding factors
        neighbor_nodes = {}  # generating the neighboring nodes of the i'th factor
        edges = np.nonzero(smat[i, :])[0]
        indices = smat[i, edges]
        for j in range(len(edges)):
            neighbor_nodes['n' + str(edges[j])] = int(indices[j])
        defg.add_factor(neighbor_nodes, np.array(factorsList[i], dtype=complex))
    return defg


def find_P(A, B, Dmax):
    """
    Finding the P matrix as in the BP truncation algorithm
    :param A: the left message
    :param B: the right message
    :param Dmax: maximal virtual bond dimension
    :return: the P matrix
    """
    A_sqrt = linalg.sqrtm(A)
    B_sqrt = linalg.sqrtm(B)

    #  Calculate the environment matrix C and its SVD
    C = np.matmul(B_sqrt, np.transpose(A_sqrt))
    u_env, s_env, vh_env = np.linalg.svd(C, full_matrices=False)

    #  Define P2
    new_s_env = cp.copy(s_env)
    new_s_env[Dmax:] = 0
    P2 = np.zeros((len(s_env), len(s_env)))
    np.fill_diagonal(P2, new_s_env)
    P2 /= np.sum(new_s_env)

    #  Calculating P = A^(-1/2) * V * P2 * U^(dagger) * B^(-1/2)
    P = np.matmul(np.transpose(np.linalg.inv(A_sqrt)),
                  np.matmul(np.transpose(np.conj(vh_env)),
                            np.matmul(P2, np.matmul(np.transpose(np.conj(u_env)), np.linalg.inv(B_sqrt)))))
    return P


def BPtruncation(tensors, weights, P, edge, smat, Dmax):
    """
    Preforming the Belief Propagation Truncation (BPT) step.
    :param tensors: the TensorNet list of tensors
    :param weights: the TensorNet list of weights
    :param P: P matrix
    :param edge: the edge we want to truncate
    :param smat: structure matrix
    :param Dmax: maximal virtual bond dimensio
    :return: updated tensors and weights lists
    """
    edgeNidxI, edgeNidxJ = getTensorsEdges(edge, smat)
    siteI, siteJ = getTensors(edge, tensors, smat)
    siteI[0] = absorbWeights(siteI[0], edgeNidxI, weights)
    siteJ[0] = absorbWeights(siteJ[0], edgeNidxJ, weights)

    # absorb the mutual edge
    siteI[0] = np.einsum(siteI[0], list(range(len(siteI[0].shape))), np.sqrt(weights[edge]), [int(siteI[2][0])], list(range(len(siteI[0].shape))))
    siteJ[0] = np.einsum(siteJ[0], list(range(len(siteJ[0].shape))), np.sqrt(weights[edge]), [int(siteJ[2][0])], list(range(len(siteJ[0].shape))))

    # reshaping
    siteI = indexPermute(siteI)
    siteJ = indexPermute(siteJ)
    i_old_shape = cp.copy(list(siteI[0].shape))
    j_old_shape = cp.copy(list(siteJ[0].shape))
    siteI[0] = rankNrank3(siteI[0])
    siteJ[0] = rankNrank3(siteJ[0])

    # contracting P with siteI and siteJ and then using SVD to generate siteI' and siteJ' and edgeWeight'
    siteI, siteJ, edgeWeight = Accordion(siteI, siteJ, P, Dmax)

    # reshaping back
    i_old_shape[1] = Dmax
    j_old_shape[1] = Dmax
    siteI[0] = rank3rankN(siteI[0], i_old_shape)
    siteJ[0] = rank3rankN(siteJ[0], j_old_shape)
    siteI = indexPermute(siteI)
    siteJ = indexPermute(siteJ)
    siteI[0] = absorbInverseWeights(siteI[0], edgeNidxI, weights)
    siteJ[0] = absorbInverseWeights(siteJ[0], edgeNidxJ, weights)

    # saving new tensors and weights
    tensors[siteI[1][0]] = siteI[0] / tensorNorm(siteI[0])
    tensors[siteJ[1][0]] = siteJ[0] / tensorNorm(siteJ[0])
    weights[edge] = edgeWeight / np.sum(edgeWeight)
    return tensors, weights


def Accordion(siteI, siteJ, P, Dmax):
    """
    Preformin the truncation step of the BPT
    :param siteI: i tensor
    :param siteJ: j tensor
    :param P: truncation P matrix
    :param Dmax: maximal virtual bond dimension
    :return: siteI, siteJ, lamda_k
    """
    # contracting two tensors i, j with P and SVD (with truncation) back
    L = siteI[0]
    R = siteJ[0]

    # contract all tensors together \theta = \sum L P R
    A = np.einsum(L, [0, 1, 2], P, [1, 3], [0, 3, 2])  # (i, Ek, Q1)
    theta = np.einsum(A, [0, 1, 2], R, [3, 1, 4], [2, 0, 3, 4])  # (Q1, i, j, Q2)

    # SVD
    R_tild, lamda_k, L_tild = truncationSVD(theta, [0, 1], [2, 3], keepS='yes', maxEigenvalNumber=Dmax)

    # reshaping R_tild and L_tild back
    R_tild_new_shape = [siteI[0].shape[2], siteI[0].shape[0], R_tild.shape[1]]  # (d, d_i, Dmax)
    R_transpose = [1, 2, 0]
    L_tild_new_shape = [L_tild.shape[0], siteJ[0].shape[0], siteJ[0].shape[2]]  # (Dmax, d_j, d)
    L_transpose = [1, 0, 2]

    # reshaping
    R_tild = np.reshape(R_tild, R_tild_new_shape)
    siteI[0] = np.transpose(R_tild, R_transpose)  # (d_i, Dmax, ...)
    L_tild = np.reshape(L_tild, L_tild_new_shape)
    siteJ[0] = np.transpose(L_tild, L_transpose)  # (d_j, Dmax, ...)
    return siteI, siteJ, lamda_k


def AnB_calculation(defg, siteI, siteJ, node_Ek):
    """
    Calculate the A, B messages for the BPT step.
    :param defg: the double-edge factor graph
    :param siteI: the TensorNet i^th tensor
    :param siteJ: the TensorNet j^th tensor
    :param node_Ek: the defg mutual node between factors I,J
    :return: A, B messages
    """
    A = defg.f2n_message_BPtruncation('f' + str(siteI[1][0]), node_Ek, defg.messages_n2f, cp.copy(siteI[0]))
    B = defg.f2n_message_BPtruncation('f' + str(siteJ[1][0]), node_Ek, defg.messages_n2f, cp.copy(siteJ[0]))
    return A, B



