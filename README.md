# shape_optimization_DRL_prep

The code in this repository presents a case of shape optimization for aerodynamics using DRL (Deep Reinforcement Learning).
It is based on Tensorforce for the DRL components, and on Fenics for the CFD computations. The corresponding paper is here : https://arxiv.org/abs/1908.09885. If you use this code for your research, please consider citing this paper.

## What is in the repo

- The parameterization of the DRL problem can be done in ```parametered_env.py```
- If you wish to modify the reward computation, you should check function ```compute_reward``` in ```environment.py```
- The ```reset``` folder contains the initial shape that is loaded as initial state at the beginning of each episode. As of now, only the 4-points initial shape is present. You can generate more initial cylinders with different point numbers using the ```generate_shape.py``` file
- The parallel learning generates copies of the environment in separate folders. To concatenate all the results in a linear history, you can run ```sort_envs.py``` periodically. All the results will be summed up in a ```sorted_envs``` folder. You should be aware that the "sorted numbering" of a shape can change from one sorting run to another
- The CFD solver is contained in the ```fenics_solver.py``` file

## Installation and requirements

This project relies on several external libraries. You will need to install Fenics (https://fenicsproject.org/download/) and gmsh (```apt-get install gmsh```).

Install the tensorforce version present in the repo:

```cd tensorforce; pip3 install -e .; cd ..```

Then install the required python modules using ```pip3```:

```pip3 install -r requirements.txt```

## How to start learning

It is recommended to use multiple terminals, or a multiplexer like ```tmux```.
First, start the servers using the following command:

```python3 launch_servers.py -n 64 -p 1111```

The ```-n``` argument corresponds to the number of available cores on your machine, while ```-p``` must be an available port (any four-digit number should be ok). Once all servers are ready, start learning:

```python3 launch_parallel_training.py -p 1111 -n 64```


