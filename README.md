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


Below are some result of ground-state energy per-site simulated with the Simple Update algorithm over Antiferromagnetic Heisenberg (AFH) Chain, Star, PEPS and Cube Tensor Networks. The AFH Hamiltonian can be written as

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/a6fe555d0e211c63ed33d4bff13ceff5fe57bbe9/assets/hamiltonian_eq.png" width="" height="100">



<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/88b9640ad702a74be625b3a0ca0069d6876fc137/assets/Tensor_Networks_diagrams.png" width="" height=""> 

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/441255a141b28806b72d42ba6c2b4525261e1eae/assets/chain_star_peps_cube.png" width="" height="">



<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/8e87ec1796e62a13c00c77261f04d68d2350443f/assets/Ising_model.png" width="" height="350">


## In Progress....
