# Generic imports
import os
import numpy

# Imports with probable installation required
try:
    import tensorforce
except ImportError:
    print('*** Missing required packages, I will install them for you ***')
    os.system('pip3 install tensorforce')
    import tensorforce

from tensorforce.agents    import PPOAgent
from tensorforce.execution import Runner

# Custom imports
from environment import *

# Parameters not going into the environment
learning_frequency      = 50
batch_size              = learning_frequency
learning_rate           = 1.0e-3
gae_lambda              = 0.97
clipping_ratio          = 0.2
entropy                 = 0.01
model_dir               = '.'

# Define environment
def resume_env():
    # Environment parameters
    reset_dir               = 'reset/4'
    nb_pts_to_move          = 4
    pts_to_move             = [0,1,2,3]
    nb_ctrls_per_episode    = 0
    nb_episodes             = 50000
    max_deformation         = 3.0
    restart_from_cylinder   = True
    replace_shape           = True
    comp_dir                = '.'
    restore_model           = False
    saving_model_period     = 10
    cfl                     = 0.5
    reynolds                = 100.0
    output_vtu              = True
    shape_h                 = 1.0
    domain_h                = 1.0
    cell_limit              = 50000
    xmin                    =-15.0
    xmax                    = 30.0
    ymin                    =-15.0
    ymax                    = 15.0
    final_time              = 2.0*(xmax-xmin)

    # Define environment
    environment=env(nb_pts_to_move, pts_to_move,
                    nb_ctrls_per_episode, nb_episodes,
                    max_deformation,
                    restart_from_cylinder,
                    replace_shape,
                    comp_dir,
                    restore_model,
                    saving_model_period,
                    final_time, cfl, reynolds,
                    output_vtu,
                    shape_h, domain_h,
                    cell_limit,
                    reset_dir,
                    xmin, xmax, ymin, ymax)

    return(environment)
