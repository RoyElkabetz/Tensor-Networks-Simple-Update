# Tensor Networks Simple-Update
This repo contains an implementation of the Simple-Update Tensor Network algorithm as described in the paper - A universal tensor network algorithm for any infinite lattice by  Saeed S. Jahromi and Roman Orus.

arXiv link to the paper - [A universal tensor network algorithm for any infinite lattice](https://arxiv.org/abs/1808.00680)

Simple Update (SU) is a Tensor Networks algorithm used for finding ground-state Tensor Network representations of gapped local Hamiltonians. It is the most efficient and the least accurate Tensor Networks algorithm. However, it is able to capture many interesting non-trivial phenomena in nD quantum spin-lattice physics. The algorithm is based on an imaginary time evolution scheme, where the ground-state of a given Hamiltonian can be computed using the next relation

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/ecb7aaf79fadf3fcb1a95a01d8866a740ab97797/assets/imaginary_time_evolution_ground_state.png" width="200" height="">

Then, in order to actually use the time evolution method in Tensor Networks we need to break down the time evolution operator into local terms. We do that using the [Suzuki-Trotter expansion](https://en.wikipedia.org/wiki/Time-evolving_block_decimation#The_Suzuki%E2%80%93Trotter_expansion). Specifically for Projected Entangled Pair States (PEPS) Tensor Networks, each local term is operating on a single pair of tensors. The approximation of breaking the time evolution operator is as follows

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/75243644b7df135e982fef5cb977f477ef10946c/assets/imaginary_time_evolution_operator.png" width="200" height="">


Here is a full step-by-step illustrated description of the algorithm. 

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/00890625f92dea318d2af588de48ba7d7a660fad/assets/simple_update_algorithm.png" width="1000" height="">


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

## Simulations
### Spin-1/2 Antiferromagnetic Heisenberg (AFH) model

Below are some result of ground-state energy per-site simulated with the Simple Update algorithm over AFH Chain, Star, PEPS and Cube Tensor Networks. The AFH Hamiltonian can be written as

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/a6fe555d0e211c63ed33d4bff13ceff5fe57bbe9/assets/hamiltonian_eq.png" width="" height="60">

In the case of the Star tensor network lattice the AFH Hamiltonian is composite of two part corresponds to different type of edges (see paper in the link above).
The Chain, Star, PEPS and Cube infinite Tensor Networks are illustrated in the next figure.

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/88b9640ad702a74be625b3a0ca0069d6876fc137/assets/Tensor_Networks_diagrams.png" width="1000" height=""> 


Here are the ground-state energy per-site simulations for the Tensor Networks diagrams above

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/5978bb7b014c41a23fed8996999f07ae1ec58edb/assets/chain_star_peps_cube_plots.png" width="1000" height="">

### Quantum Ising Model on a 2D Spin-1/2 Lattice
Next is the quantum Ising model simulated on a 2D lattice with a transverse field. Its Hamiltonian is given by

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/2c4d0137d856b282f3ba4bfa0e81ef5a4be67e99/assets/ising_transverse_field.png" width="" height="100">

In the plots below oone can see the simulated x, z magnetization (per-site) along with the simulated energy (per-site). We see that the SU is manage to extract the phase transition of the model around h=3.2.

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/8e87ec1796e62a13c00c77261f04d68d2350443f/assets/Ising_model.png" width="1000" height="">

### TODO

- [ ] Add more terms to the Hamiltonian.
- [ ] Add trivial simple update algorithm for canonical states computation.
- [ ] Add Manim illustration of a simple update run

## In Progress....
