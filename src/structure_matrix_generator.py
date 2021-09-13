"""
This module contains functions for structure matrix generation. Still in progress, may have some bugs !
"""

import numpy as np


def infinite_structure_matrix_by_name(name: str):
    """
    A dictionary of iPEPS structure matrices as written in the paper "A universal tensor network algorithm for any infinite lattice".
    """
    structure_matrix = {'chain': np.array([[1, 2],
                                           [1, 2]]),

                        'peps': np.array([[1, 2, 3, 4, 0, 0, 0, 0],
                                          [1, 2, 0, 0, 3, 4, 0, 0],
                                          [0, 0, 1, 2, 0, 0, 3, 4],
                                          [0, 0, 0, 0, 1, 2, 3, 4]]),

                        'star': np.array([[1, 2, 3, 0, 0, 0, 0, 0, 0],
                                          [1, 0, 0, 2, 3, 0, 0, 0, 0],
                                          [0, 1, 0, 2, 0, 3, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 1, 2, 3, 0],
                                          [0, 0, 0, 0, 1, 0, 2, 0, 3],
                                          [0, 0, 1, 0, 0, 0, 0, 2, 3]]),

                        'cube': np.array([[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [1, 2, 0, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 5, 6]]),

                        'triangle': np.array([[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0, 0, 5, 6, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 5, 6, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 5, 0, 6],
                                             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 4, 0, 5, 6]]),
    }
    return structure_matrix[name]


def peps_square_periodic_boundary_conditions(side: np.int):
    """
    returns a structure matrix of a square lattice tensor network with periodic boundary conditions of shape (side, side).
    The total nmber of tensors in the network would be side^2.
    : param side The hight and width values.
    """
    n_tensors = np.int(np.square(side))
    structure_matrix = np.zeros((n_tensors, 2 * n_tensors), dtype=np.int)
    for i in range(n_tensors):
        structure_matrix[i, i] = 4
        structure_matrix[i, i + n_tensors] = 1
        if i % side == 0:
            structure_matrix[i, i + side - 1] = 2
        else:
            structure_matrix[i, i - 1] = 2
        if i >= side:
            structure_matrix[i, n_tensors - side + i] = 3
        else:
            structure_matrix[i, 2 * n_tensors - side + i] = 3
    return structure_matrix


def peps_rectangular_open_boundary_conditions(height: np.int, width: np.int):
    """
    returns a structure matrix of a rectangular lattice tensor network with open (non-periodic) boundary conditions of shape  (height x width).
    The total nmber of tensors in the network would be hight x width.
    : param hight The hight of the tensor network.
    : param width The width of the tensor network.
    """
    # edge = (node_a i, node_a j, node_a l, node_b i, node_b j, node_b l)
    edge_list = []
    for i in range(height):
        for j in range(width):
            if i < height - 1:
                edge_list.append((i, j, 4, i + 1, j, 2))
            if j < width - 1:
                edge_list.append((i, j, 3, i, j + 1, 1))
    structure_matrix = np.zeros(shape=[height * width, len(edge_list)], dtype=np.int)

    for edge_idx, edge in enumerate(edge_list):
        node_a_idx = np.ravel_multi_index([edge[0], edge[1]], (height, width))
        node_b_idx = np.ravel_multi_index([edge[3], edge[4]], (height, width))
        structure_matrix[node_a_idx, edge_idx] = edge[2]
        structure_matrix[node_b_idx, edge_idx] = edge[5]

    for i in range(structure_matrix.shape[0]):
        row = structure_matrix[i, np.nonzero(structure_matrix[i, :])[0]]
        new_row = np.array(range(1, len(row) + 1))
        order = np.argsort(row)
        new_row = new_row[order]
        structure_matrix[i, np.nonzero(structure_matrix[i, :])[0]] = new_row
    return structure_matrix


