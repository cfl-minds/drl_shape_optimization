# Copyright 2018 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tensorforce.agents import DRLAgent
from tensorforce.core.models import QNstepModel


class DQNNstepAgent(DRLAgent):
    """
    DQN n-step agent.
    """

    def __init__(
        self,
        states,
        actions,
        memory,
        parallel_interactions=1,
        buffer_observe=1000,
        seed=None,
        scope='dqn-nstep',
        device=None,
        saver=None,
        summarizer=None,
        execution=None,
        exploration=None,
        variable_noise=None,
        states_preprocessing=None,
        reward_preprocessing=None,
        update_mode=None,
        optimizer=None,
        discount=None,
        network='auto',
        distributions=None,
        entropy_regularization=None,
        target_sync_frequency=10000,
        target_update_weight=None,
        double_q_model=False,
        huber_loss=None
    ):
        """
        Initializes the DQN n-step agent.

        Args:
            update_mode (spec): Update mode specification, with the following attributes:
                - unit: 'episodes' if given (default: 'episodes').
                - batch_size: integer (default: 10).
                - frequency: integer (default: batch_size).
            memory (spec): Memory specification, see core.memories module for more information
                (default: {type='latest', include_next_states=true, capacity=1000*batch_size}).
            optimizer (spec): Optimizer specification, see core.optimizers module for more
                information (default: {type='adam', learning_rate=1e-3}).
            target_sync_frequency (int): Target network sync frequency (default: 10000).
            target_update_weight (float): Target network update weight (default: 1.0).
            double_q_model (bool): Specifies whether double DQN mode is used (default: false).
            huber_loss (float): Huber loss clipping (default: none).
        """
        super().__init__(
            states=states, actions=actions, parallel_interactions=parallel_interactions,
            buffer_observe=buffer_observe, seed=seed
        )

        # Update mode
        if update_mode is None:
            update_mode = dict(unit='episodes', batch_size=10)
        elif 'unit' in update_mode:
            assert update_mode['unit'] == 'episodes'
        else:
            update_mode['unit'] = 'episodes'

        # Memory
        if isinstance(memory, int):
            memory = dict(type='latest', include_next_states=True, capacity=memory)
        else:
            assert memory['include_next_states']

        # Optimizer
        if optimizer is None:
            optimizer = dict(type='adam', learning_rate=1e-3)

        self.model = QNstepModel(
            # Model
            states=self.states_spec, actions=self.actions_spec, scope=scope, device=device,
            saver=saver, summarizer=summarizer, execution=execution,
            parallel_interactions=self.parallel_interactions, buffer_observe=self.buffer_observe,
            exploration=exploration, variable_noise=variable_noise,
            states_preprocessing=states_preprocessing, reward_preprocessing=reward_preprocessing,
            # MemoryModel
            update_mode=update_mode, memory=memory, optimizer=optimizer, discount=discount,
            # DistributionModel
            network=network, distributions=distributions,
            entropy_regularization=entropy_regularization,
            # QModel
            target_sync_frequency=target_sync_frequency, target_update_weight=target_update_weight,
            double_q_model=double_q_model, huber_loss=huber_loss
        )
