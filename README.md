# Tensor Networks Simple-Update (SU) Algorithm
> This python package contains an implementation of the Simple-Update Tensor Network algorithm as described in the paper - A universal tensor network algorithm for any infinite lattice by Saeed S. Jahromi and Roman Orus [1].

 
### Installation
```bash
pip3 install tnsu
```

### Documentation
For details about the `tnsu` package, see the github repo in [this link](https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update).



## Simple Update
Simple Update (SU) is a Tensor Networks (TN) algorithm used for finding ground-state Tensor Network representations of [gapped local Hamiltonians](https://en.wikipedia.org/wiki/Gapped_Hamiltonian). It is the TN most efficient and least accurate algorithm for computing ground states. However, it is able to capture many interesting non-trivial phenomena in n-D quantum spin-lattice physics. The algorithm is based on an Imaginary Time-Evolution (ITE) scheme, where the ground-state of a given Hamiltonian can be obtained following the next relation

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/assets/imaginary_time_evolution_ground_state.png?raw=true" width="300" height="">

In order to actually use the time-evolution method in TN we need to break down the time-evolution operator into local terms. We do that with the help of the [Suzuki-Trotter expansion](https://en.wikipedia.org/wiki/Time-evolving_block_decimation#The_Suzuki%E2%80%93Trotter_expansion). Specifically for Projected Entangled Pair States (PEPS) TN, each local term corresponds to a single pair of tensors. The Suzuki-Trotter approximation steps for ITE are as follows

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/assets/imaginary_time_evolution_operator.png?raw=true" width="300" height="">

where finally,

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/assets/ITE_local_gate.png?raw=true" width="200" height="">

When performing the ITE scheme, the TN virtual bond dimension increases, therefore, after every few ITE iterations we need to truncate the bond dimensions so the number of parameters in the tensor network state would stay bounded. The truncation step is implemented via a [Singular Value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition) step. A full step-by-step illustrated description of the Simple Update algorithm (which is based on the ITE scheme) is depicted below. 

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/assets/simple_update_algorithm.png?raw=true" width="1000" height="">

For a more comprehensive explanation of the algorithm, the interested reader should check out [1].

## The Code


The [`src.tnsu`](/src/tnsu) folder contains the source code for this project

| #   | file                                         | Subject             | 
|:----:|------------------------------------------------|:-----------------:|
| 1   | `tensor_network.py`                   | a Tensor Network class object which tracks the tensors, weights and their connectivity| 
| 2   | `simple_update.py`         | a Tensor Network Simple-Update algorithm class, which gets as an input a `TensorNetwork` object and perform a simple-update run on it using Imaginary Time Evolution. | 
| 3  | `structure_matrix_constructor.py`         | Contains a dictionary of common iPEPS structure matrices and also functionality construction of 2D square and rectangular lattices structure matrices (**still in progress**).
| 4  | `examples.py`         | Few scripts for loading a tensor network state from memory and a full Antiferromagnetic Heisenberg model PEPS experiment.|
| 5  | `ncon.py`         | A module for tensors contraction in python copied from the [ncon](https://github.com/mhauru/ncon) github repository.|
| 6  | `utils.py`         | A general utility module.|



## Examples

### Example 1: Spin 1/2 2D star lattice iPEPS Antiferromagnetic Heisenberg model simulation

Importing files
```python
import numpy as np
from tnsu.tensor_network import TensorNetwork
import tnsu.simple_update as su
import structure_matrix_constructor as stmc
```
First let us get the iPEPS star structure matrix

```python
smat = stmc.infinite_structure_matrix_dict('star')
```

Next we initialize a random Tensor Network with a virtual bond dimension of size 2 and a physical spin dimension also of size 2
```python
tensornet = TensorNetwork(structure_matrix=smat, virtual_size=2, spin_dim=2)
```

Then, set up the spin 1/2 operators and the simple update class parameters 
```python
# pauli matrices
pauli_x = np.array([[0, 1],
                    [1, 0]])
pauli_y = np.array([[0, -1j],
                    [1j, 0]])
pauli_z = np.array([[1., 0],
                    [0, -1]])
# ITE time constants
dts = [0.1, 0.01, 0.001, 0.0001, 0.00001]

# Local spin operators
s = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]

# The Hamiltonian's 2-body interaction constants 
j_ij = [1., 1., 1., 1., 1., 1.]

# The Hamiltonian's 1-body field constant
h_k = 0.

# The field-spin operators (which are empty in that example)
s_k = []

# The maximal virtual bond dimension (used for SU truncation)
d_max = 2
```

Now, we initialize the simple update class
```python
star_su = su.SimpleUpdate(tensor_network=tensornet, 
                          dts=dts, 
                          j_ij=j_ij, 
                          h_k=h_k, 
                          s_i=s, 
                          s_j=s, 
                          s_k=s_k, 
                          d_max=d_max, 
                          max_iterations=200, 
                          convergence_error=1e-6, 
                          log_energy=False,
                          print_process=False)
```

and run the algorithm
```python
star_su.run()
```

It is also possible to compute a single and double site expectation values like energy, magnetizatoin etc, with the following
```python
energy_per_site = star_su.energy_per_site()
z_magnetization_per_site = star_su.expectation_per_site(operator=pauli_z / 2)
```

or manually calculating single and double site reduced-density matrices and expectations following the next few lines of code
```python
tensor = 0
edge = 1
tensor_pair_operator = np.reshape(np.kron(pauli_z / 2, pauli_z / 2), (2, 2, 2, 2))
star_su.tensor_rdm(tensor_index=tensor)
star_su.tensor_pair_rdm(common_edge=edge)
star_su.tensor_expectation(tensor_index=tensor, operator=pauli_z / 2)
star_su.tensor_pair_expectation(common_edge=edge, operator=tensor_pair_operator)
```

### Example 2: The Trivial Simple-Update Algorithm
The trivial SU algorithm is equivalent to the SU algorithm without the ITE and truncation steps; it only consists of consecutive SVD steps over each TN edge (the same as contracting ITE gate with zero time-step). The trivial-SU algorithm's fixed point corresponds to a canonical representation of the tensor network representations we started with. A tensor network canonical representation is strongly related to the Schmidt Decomposition operation over all the tensor network's edges, where for a tensor networks with no loops (tree-like topology) each weight vector in the canonical representation corresponds to the Schmidt values of partitioning the network into two distinct networks along that edge. When the given tensor network has loops in it, it is no longer possible to partition the network along a single edge into to distinguished parts. Therefore, the weight vectors are no longer equal to the Schmidt values but rather become some general approximation of the tensors' environments in the network. A very interesting property of the trivial simple update algorithm is that it is identical to the [Belief Propagation (BP)](https://en.wikipedia.org/wiki/Belief_propagation) algorithm. The Belief Propagation (BP) algorithm is a famous iterative-message-passing algorithm in the world of Probabilistic Graphical Models (PGM), where it is used as an approximated inference tool. For a detailed description about the duality between the trivial-Simple-Update and the Belief Propagation algorithm see Refs [3][4].

In order to implement the trivial-SU algorithm we can initialize the simple update class with zero time step as follows
```python
su.SimpleUpdate(tensor_network=tensornet, 
                dts=[0], 
                j_ij=j_ij, 
                h_k=0, 
                s_i=s, 
                s_j=s, 
                s_k=s_k, 
                d_max=d_max, 
                max_iterations=1000, 
                convergence_error=1e-6, 
                log_energy=False,
                print_process=False)
```
then, the algorithm will run 1000 iteration or until the maximal L2 distance between temporal consecutive weight vectors will be smaller then 1e-6.


There are more fully-written examples in the [`notebooks`](/notebooks) folder.

### List of Notebooks
The notebooks are currently in progress...

| #   | file            | Subject                                         | Colab             | Nbviewer               |
|:----:|:--------------:|:------------------------------------------------:|:-----------------:|:---------------------:|
| 1   | `ipeps_energy_simulations.ipynb` | Computing ground-state energies of iPEPS Tensor Networks   | [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/notebooks/ipeps_energy_simulations.ipynb)        | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/notebooks/ipeps_energy_simulations.ipynb)|
| 2   | `Quantum_Ising_Model_Phase_Transition.ipynb` | Simulating the phase transition of the Quantum Transverse Field Ising model  | [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/notebooks/Quantum_Ising_Model_Phase_Transition.ipynb)        | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/notebooks/Quantum_Ising_Model_Phase_Transition.ipynb)|
| 3   | `Triangular_2d_lattice_BLBQ_Spin_1_simulation.ipynb` | Spin-1 BLBQ tringular 2D lattice phase transition   | [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/notebooks/Triangular_2d_lattice_BLBQ_Spin_1_simulation.ipynb)        | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/notebooks/Triangular_2d_lattice_BLBQ_Spin_1_simulation.ipynb)|


## Simulations
### Spin-1/2 Antiferromagnetic Heisenberg (AFH) model

Below are some result of ground-state energy per-site simulated with the Simple Update algorithm over AFH Chain, Star, PEPS and Cube tensor networks. The AFH Hamiltonian is given by

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/assets/hamiltonian_eq.png?raw=true" width="" height="60">

In the case of the Star tensor network lattice the AFH Hamiltonian is composite of two parts which corresponds to different type of edges (see [1]).
The Chain, Star, PEPS and Cube infinite tensor networks are illustrated in the next figure.

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/assets/Tensor_Networks_diagrams.png?raw=true" width="1000" height=""> 


Here are the ground-state energy per-site vs inverse virtual bond-dimension simulations for the tensor networks diagrams above

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/assets/chain_star_peps_cube_plots.png?raw=true" width="1000" height="">

### Quantum Ising Model on a 2D Spin-1/2 Lattice
Next, we simulated the quantum Ising model on a 2D lattice with a transverse magnetic field. Its Hamiltonian is given by

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/assets/ising_transverse_field.png?raw=true" width="" height="100">

In the plots below one can see the simulated x, z magnetization (per-site) along with the simulated energy (per-site). We see that the SU algorithm is able to extract the phase transition of the model around h=3.2.

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/assets/Ising_model.png?raw=true" width="1000" height="">

### Spin-1 Simulation of a Bilinear-Biquadratic Heisenberg model on a star 2D lattice

Finally we simulated the BLBQ Hamiltonian which is given by the next equation

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/assets/BLBQ_hamiltonian.png?raw=true" width="300" height="">

notice that for 0 radian angle, this model coincides with the original AFH model. The energy, magnetization and Q-norm as a function of the angle for different bond dimension are plotted below. We can see that the simple-update algorithm is having a hard time to trace all the phase transitions of this model. However, we notice that for larger bond dimensions it seems like it captures the general behavior of the model's phase transition. For a comprehensive explanation and results (for triangular lattice see Ref [2])

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/assets/BLBQ_model_simulation_star.png?raw=true" width="1000" height="">

## References
- [[1]](https://arxiv.org/abs/1808.00680) Saeed S. Jahromi, and Roman Orus - "A universal tensor network algorithm for any infinite lattice" (2019)
- [[2]](https://arxiv.org/abs/1805.00354) Ido Niesen, Philippe Corboz - "A ground state study of the spin-1 bilinear-biquadratic Heisenberg model on the triangular lattice using tensor networks" (2018)
- [[3]](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023073) Roy Alkabetz and Itai Arad - "Tensor networks contraction and the belief propagation algorithm" (2020)
- [[4]](https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/0f7341e175c7f1fdd6723855749800927d37ebc8/theory/MSc_Thesis_Roy_Elkabetz.pdf) Roy Elkabetz - "Using the Belief Propagation algorithm for finding Tensor Networks approximations of many-body ground states" (2020)


## Contact

Roy Elkabetz - [elkabetzroy@gmail.com](mailto:elkabetzroy@gmail.com)

## Citation

To cite this repository in academic works or any other purpose, please use the following BibTeX citation:
```@software{tnsu,
    author = {Elkabetz, Roy},
    title = {{tnsu: A python package for Tensor Networks Simple-Update simulations}},
    url = {https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update},
    version = {1.0.0},
    year = {2022}
}
```

