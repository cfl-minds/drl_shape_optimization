# shape_optimization_DRL_prep

The code in this repository presents a case of shape optimization for aerodynamics using DRL (Deep Reinforcement Learning).
It is based on Tensorforce for the DRL components, and on Fenics for the CFD computations. The corresponding paper is here : https://arxiv.org/abs/1908.09885. If you use this code for your research, please consider citing this paper.

## What is in the repo

- The parameterization of the DRL problem can be done in ```parametered_env.py```.

- If you wish to modify the reward computation, you should check the function ```compute_reward``` in ```environment.py```.

- The ```reset``` folder contains the initial shape that is loaded as initial state at the beginning of each episode. As of now, only the 4-points initial shape is present. You can generate more initial cylinders with different point numbers using the ```generate_shape.py``` file.

- The parallel learning generates copies of the environment in separate folders. To concatenate all the results in a linear history, you can run ```sort_envs.py``` periodically. All the results will be summed up in a ```sorted_envs``` folder. You should be aware that the "sorted numbering" of a shape can change from one sorting run to another.

- The CFD solver is contained in the ```fenics_solver.py``` file.

## Installation and requirements

### Method 1 (discouraged): installing by hand

- This project relies on several external libraries. You will need to install Fenics (https://fenicsproject.org/download/) and gmsh, in the right versions (Gmsh 3.0.6, dolfin/fenics 2018.1.0), on your platform. This may be tricky to get given the complex dependency graphs of both these packages.

In addition, you will need the following packages that are provided / defined in the repo (you may need to run these with sudo rights):

- Install the tensorforce version present in the repo:

```cd tensorforce; pip3 install -e .; cd ..```

- Then install the required python modules using ```pip3```:

```pip3 install -r requirements.txt```

### Method 2 (recommended): using the docker container provided

Get the docker container at: TODO-URL-release

Load the docker container, and start an interactive session sharing your cwd to allow exchange of information in and out of the container:

```
sudo docker load < shape_2d_2020_12.tar  # load the image into docker
sudo docker run -ti --name shape2dopt -v $(pwd):/home/fenics/shared shape_2d_2020_12:latest  # spin an interactive container (named shape2dopt) out of the image, sharing the cwd for allowing exchange of information
  at this stage you get a prompt inside the container; exit the container immediately
sudo docker start shape2dopt  # container was stopped when exiting, start it again
sudo docker exec -ti -u fenics shape2dopt /bin/bash -l  # get an interactive terminal inside the re-started container
  you are now in an interactive session inside the container, where the /shared folder is a mirror of your cwd, and /local contains the code, with all packages and dependencies availalbe
```

In addition, once you are finished working, you can stop the whole container by using ```sudo docker container stop shape2dopt```

The code contained in this repo is included already in the container, at: ```/home/fenics/local/shape_optimization_DRL_prep```. The source code for performing learning is available at: ```/home/fenics/local/shape_optimization_DRL_prep/src```. All necessary dependencies and packages in the correct version are already included, no more setup is needed.

## How to start learning

It is recommended to use multiple terminals, or a multiplexer like ```tmux```, in order to manage the simulation servers vs the training client.

- First, start the servers (for example in a tmux pane) using the following command:

```python3 launch_servers.py -n 4 -p 1111```

The ```-n``` argument corresponds to the number of available cores on your machine, while ```-p``` must be an available port (any four-digit number should be ok, feel free to change it if 1111 is taken on your machine).

- Once all servers are ready (confirmed by the mention ```all processes started, ready to serve...```), start the training (for example in a new tmux pane):

```python3 launch_parallel_training.py -p 1111 -n 4```

Note that the number of servers and the first port value must match that used for starting the simulation servers.

- At this stage, the DRL shape optimization is now working.

