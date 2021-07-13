# Tensor Networks Simple-Update
This repo contains an implementation of the Simple-Update Tensor Network algorithm as described in the paper - A universal tensor network algorithm for any infinite lattice by  Saeed S. Jahromi and Roman Orus.

DOI:	10.1103/PhysRevB.99.195105

arXiv link to paper - [A universal tensor network algorithm for any infinite lattice](https://arxiv.org/abs/1808.00680)

## The Code



The folder [`src`](/src) is he heart of this project, here is a description of each file

| #   | file                                         | Subject             | 
|:----:|------------------------------------------------|:-----------------:|
| 1   | `TensorNetwork.py`                   | This is a Tensor Network class object which tracks the tensors, weights and their connectivity| 
| 2   | `SimpleUpdate.py`         | This is a Tensor Network Simple Update algorithm class which get as an input a TensorNetwork object and perform a simple update run on it. | 

### List of Notebooks

| #   | Subject                                         | Colab             | Nbviewer               |
|:----:|------------------------------------------------|:-----------------:|:---------------------:|
| 1   | Paper results reconstruction                   | [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/univeral_tensor_network_paper__reconstruction.ipynb#scrollTo=x9gTThCjbrzm)        | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/univeral_tensor_network_paper__reconstruction.ipynb)|


Here are some result of ground-state energy per-site calculated with Simple Update over Chain, Star, PEPS and Cube Tensor Networks

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/cc715d4b92f4610bdc612dcd68771f624a2357e7/assets/Tensor_Networks_diagrams.png" width="400" height=""> <img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/f9dbfc21f0f724469feef5f4ac23448149c29aa3/assets/chain_star_peps_cube.png" width="400" height="">

## In Progress....
