# Tensor Networks Simple-Update (SU) Algorithm
> This repo contains an implementation of the Simple-Update Tensor Network algorithm as described in the paper - A universal tensor network algorithm for any infinite lattice by  Saeed S. Jahromi and Roman Orus.

> arXiv link to the paper - [A universal tensor network algorithm for any infinite lattice](https://arxiv.org/abs/1808.00680)

## Simple Update
Simple Update (SU) is a Tensor Networks algorithm used for finding ground-state Tensor Network representations of [gapped local Hamiltonians](https://en.wikipedia.org/wiki/Gapped_Hamiltonian). It is the most efficient and the least accurate Tensor Networks algorithm. However, it is able to capture many interesting non-trivial phenomena in nD quantum spin-lattice physics. The algorithm is based on an Imaginary Time-Evolution (ITE) scheme, where the ground-state of a given Hamiltonian can be obtained following the next relation

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/ecb7aaf79fadf3fcb1a95a01d8866a740ab97797/assets/imaginary_time_evolution_ground_state.png" width="300" height="">

In order to actually use the time-evolution method in Tensor Networks we need to break down the time-evolution operator into local terms. We do that with the [Suzuki-Trotter expansion](https://en.wikipedia.org/wiki/Time-evolving_block_decimation#The_Suzuki%E2%80%93Trotter_expansion). Specifically for Projected Entangled Pair States (PEPS) Tensor Networks, each local term will be operating on a single pair of tensors. The Suzuki-Trotter approximation steps of breaking the time evolution operator is as follows

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/75243644b7df135e982fef5cb977f477ef10946c/assets/imaginary_time_evolution_operator.png" width="300" height="">

where finally

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/4b0b3dde85654d82c69542a49f1e86c156d4c140/assets/ITE_local_gate.png" width="300" height="">

When performing the ITE scheme, the Tensor Network virtual bond dimension increases, therefore after every few time steps we need to truncate it so the number of parameters in the tensor network state would stay bounded. This truncation step is implemented via a [Singular Value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition) step. A full step-by-step illustrated description of the algorithm is depicted below. 

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/00890625f92dea318d2af588de48ba7d7a660fad/assets/simple_update_algorithm.png" width="1000" height="">


## The Code


The [`src`](/src) folder contains the code of this project

| #   | file                                         | Subject             | 
|:----:|------------------------------------------------|:-----------------:|
| 1   | `TensorNetwork.py`                   | a Tensor Network class object which tracks the tensors, weights and their connectivity| 
| 2   | `SimpleUpdate.py`         | a Tensor Network Simple-Update algorithm class which get as an input a `TensorNetwork` object and perform a simple-update run on it using imaginary-time-evolution. | 
| 3  | `structure_matrix_generator.py`         | This file containes a dictionary of common iPEPS structure matrices and also some functions for 2D square and rectangular lattices structure matrices (**still in progress**)

### Example: Spin 1/2 2D star lattice iPEPS Antiferromagnetic Heisenberg model simulation

Importing files
```python
from TensorNetwork import TensorNetwork
import SimpleUpdate as su
import structure_matrix_generator as stmg
```
Get the iPEPS star structure matrix

```python
smat = stmg.infinite_structure_matrix_by_name('star')
```

Initialize a random Tensor Network with virtual bond dimension of size 2 and physical spin dimension also of size 2
```python
tensornet = TensorNetwork(structure_matrix=smat, virtual_size=2, spin_dim=2)
```

Then, set the spin 1/2 operators and the simple update class parameters 
```python
# pauli matrices
pauli_x = np.array([[0, 1],
                    [1, 0]])
pauli_y = np.array([[0, -1j],
                    [1j, 0]])
pauli_z = np.array([[1., 0],
                    [0, -1]])
# su parameters                    
dts = [0.1, 0.01, 0.001, 0.0001, 0.00001]
s = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
j_ij = [1., 1., 1., 1., 1., 1.]
h_k = 0.
s_k = []
d_max = 2
```

Initialize the simple update class
```python
star_su = su.SimpleUpdate(tensor_network=tensornet, 
                          dts=dts, 
                          j_ij=j_ij, 
                          h_k=0, 
                          s_i=s, 
                          s_j=s, 
                          s_k=s_k, 
                          d_max=d_max, 
                          max_iterations=200, 
                          convergence_error=1e-6, 
                          log_energy=False,
                          print_process=False,
                          )
```

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
