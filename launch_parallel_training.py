# Generic imports
import os
import argparse
import numpy as np

# Stocketing imports
import socket
from   tensorforce_socketing.RemoteEnvironmentClient import RemoteEnvironmentClient

# Imports with probable installation required
try:
    import tensorforce
except ImportError:
    print('*** Missing required packages, I will install them for you ***')
    os.system('pip3 install tensorforce')
    import tensorforce

from tensorforce.agents    import PPOAgent
from tensorforce.execution import ParallelRunner

# Custom imports
from parametered_env import *

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
ap.add_argument("-p", "--ports-start", required=True, help="the start of the range of ports to use", type=int)
ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)

args           = vars(ap.parse_args())
number_servers = args["number_servers"]
ports_start    = args["ports_start"]
host           = args["host"]

if host == 'None': host = socket.gethostname()

num_parallel = number_servers
environment  = resume_env()
remote_envs  = []

# Generate environments
for crrt_simu in range(num_parallel):
    current_remote_env = RemoteEnvironmentClient(environment,
                                                 verbose=0,
                                                 port=ports_start + crrt_simu,
                                                 host=host)
    remote_envs.append(current_remote_env)

# Define agent
agent = PPOAgent(
    states=environment.states,
    actions=environment.actions,
    network='auto',
    parallel_interactions=num_parallel,
    # Agent
    states_preprocessing=None,
    reward_preprocessing=None,
    # MemoryModel
    update_mode=dict(
        unit='episodes',
        batch_size=batch_size,
        frequency=learning_frequency
    ),
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=10000
    ),
    # DistributionModel
    distributions=None,
    entropy_regularization=entropy,
    # PGModel
    baseline_mode='states',
    baseline=dict(
        type='network',
        network='auto'
    ),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=learning_rate
        ),
        num_steps=5
    ),
    gae_lambda=gae_lambda,
    # PGLRModel
    likelihood_ratio_clipping=clipping_ratio,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=learning_rate
    ),
    subsampling_fraction=0.2,
    optimization_steps=25
)

# Run agent
runner = ParallelRunner(agent=agent, environments=remote_envs)

# Start learning
runner.run(num_episodes=environment.nb_episodes,
           max_episode_timesteps=environment.nb_ctrls_per_episode)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
