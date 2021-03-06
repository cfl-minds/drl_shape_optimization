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

from collections import OrderedDict
from copy import deepcopy
import os

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import Module, parameter_modules
from tensorforce.core.preprocessors import PreprocessorStack


class Model(Module):
    """
    The `Model` class coordinates the creation and execution of all TensorFlow operations within a model.
    It implements the `reset`, `act` and `update` functions, which form the interface the `Agent` class
    communicates with, and which should not need to be overwritten. Instead, the following TensorFlow
    functions need to be implemented:

    * `tf_actions_and_internals(states, internals, deterministic)` returning the batch of
       actions and successor internal states.

    Further, the following TensorFlow functions should be extended accordingly:

    * `setup_placeholders()` defining TensorFlow input placeholders for states, actions, rewards, etc..
    * `setup_template_funcs()` builds all TensorFlow functions from "tf_"-methods via tf.make_template.
    * `get_variables()` returning the list of TensorFlow variables (to be optimized) of this model.

    Finally, the following TensorFlow functions can be useful in some cases:

    * `tf_preprocess(states, internals, reward)` for states/action/reward preprocessing (e.g. reward normalization),
        returning the pre-processed tensors.
    * `tf_action_exploration(action, exploration, actions)` for action postprocessing (e.g. exploration),
        returning the processed batch of actions.
    * `create_output_operations(states, internals, actions, terminal, reward, deterministic)` for further output operations,
        similar to the two above for `Model.act` and `Model.update`.
    * `tf_optimization(states, internals, actions, terminal, reward)` for further optimization operations
        (e.g. the baseline update in a `PGModel` or the target network update in a `QModel`),
        returning a single grouped optimization operation.
    """

    def __init__(
        self,
        # Model
        states, internals, actions, scope, device, saver, summarizer, execution,
        parallel_interactions, buffer_observe, exploration, variable_noise, states_preprocessing,
        reward_preprocessing
    ):
        if summarizer is None or summarizer.get('directory') is None:
            summary_labels = None
        else:
            summary_labels = summarizer.get('labels', ())

        super().__init__(name=scope, l2_regularization=None, summary_labels=summary_labels)

        # States/internals/actions specifications
        self.states_spec = states
        self.internals_spec = OrderedDict() if internals is None else internals
        for name in self.internals_spec:
            if name in self.states_spec:
                raise TensorforceError(
                    "Name overlap between internals and states: {}.".format(name)
                )
        self.actions_spec = actions
        for name in self.actions_spec:
            if name in self.states_spec:
                raise TensorforceError(
                    "Name overlap between actions and states: {}.".format(name)
                )
            if name in self.internals_spec:
                raise TensorforceError(
                    "Name overlap between actions and internals: {}.".format(name)
                )
        self.values_spec = OrderedDict(
            states=self.states_spec, internals=self.internals_spec, actions=self.actions_spec,
            terminal=dict(type='bool', shape=()), reward=dict(type='float', shape=())
        )

        # TensorFlow scope and device
        self.scope = scope
        self.device = device

        # Saver
        if saver is None or saver.get('directory') is None:
            self.saver_spec = None
        else:
            self.saver_spec = saver

        # Summarizer
        if summarizer is None or summarizer.get('directory') is None:
            self.summarizer_spec = None
        else:
            self.summarizer_spec = summarizer

        # Execution
        if execution is None or (
            execution['type'] == 'single' and execution['session_config'] is None and
            execution['distributed_spec'] is None
        ):
            self.execution_spec = None
        else:
            self.execution_spec = execution
            self.execution_type = self.execution_spec['type']
            self.session_config = self.execution_spec['session_config']
            self.distributed_spec = self.execution_spec['distributed_spec']
        # One record is inserted into these buffers when act(independent=False) method is called.
        # self.num_parallel = self.execution_spec.get('num_parallel', 1)

        # Parallel interactions
        assert isinstance(parallel_interactions, int) and parallel_interactions >= 1
        self.parallel_interactions = parallel_interactions

        # Buffer observe
        assert isinstance(buffer_observe, int) and buffer_observe >= 1
        self.buffer_observe = buffer_observe
        # from tensorforce.core.utils import CircularBuffer
        # values_spec = OrderedDict(
        #     states=self.states_spec, internals=self.internals_spec, actions=self.actions_spec
        # )
        # self.observe_buffer = self.add_module(
        #     name='observe-buffer', module=CircularBuffer, values_spec=values_spec,
        #     capacity=buffer_observe
        # )

        # Actions exploration
        assert exploration is None or isinstance(exploration, dict) or exploration >= 0.0
        exploration = 0.0 if exploration is None else exploration
        if isinstance(exploration, dict) and \
                all(name in self.actions_spec for name in exploration):
            # Different exploration per action
            self.exploration = OrderedDict()
            for name in self.actions_spec:
                if name in exploration:
                    self.exploration[name] = self.add_module(
                        name=(name + '-exploration'), module=exploration[name],
                        modules=parameter_modules
                    )
        else:
            # Same exploration for all actions
            self.exploration = self.add_module(
                name='exploration', module=exploration, modules=parameter_modules
            )

        # Variable noise
        assert variable_noise is None or isinstance(variable_noise, dict) or variable_noise >= 0.0
        variable_noise = 0.0 if variable_noise is None else variable_noise
        self.variable_noise = self.add_module(
            name='variable-noise', module=variable_noise, modules=parameter_modules
        )

        # States preprocessing
        self.states_preprocessing = OrderedDict()
        if states_preprocessing is None:
            pass

        elif isinstance(states_preprocessing, dict) and \
                all(name in self.states_spec for name in states_preprocessing):
            # Different preprocessing per state
            for name, state_spec in self.states_spec.items():
                if name in states_preprocessing:
                    self.states_preprocessing[name] = preprocessing = self.add_module(
                        name=(name + '-preprocessing'), module=states_preprocessing[name],
                        modules=preprocessor_modules, default=PreprocessorStack,
                        shape=state_spec['shape']
                    )
                    state_spec['unprocessed_shape'] = state_spec['shape']
                    state_spec['shape'] = preprocessing.processed_shape(shape=state_spec['shape'])

        else:
            # Same preprocessing for all states
            for name, state_spec in self.states_spec.items():
                self.states_preprocessing[name] = preprocessing = self.add_module(
                    name=(name + '-preprocessing'), module=states_preprocessing,
                    modules=preprocessor_modules, default=PreprocessorStack,
                    shape=state_spec['shape']
                )
                state_spec['unprocessed_shape'] = state_spec['shape']
                state_spec['shape'] = preprocessing.processed_shape(shape=state_spec['shape'])

        # Reward preprocessing
        if reward_preprocessing is None:
            self.reward_preprocessing = None

        else:
            self.reward_preprocessing = self.add_module(
                name='reward-preprocessing', module=reward_preprocessing,
                modules=preprocessor_modules, default=PreprocessorStack, shape=()
            )
            if self.reward_preprocessing.processed_shape(shape=()) != ():
                raise TensorforceError("Invalid reward preprocessing!")

        # Register global tensors
        for name, spec in self.states_spec.items():
            Module.register_tensor(name=name, spec=spec, batched=True)
        for name, spec in self.internals_spec.items():
            Module.register_tensor(name=name, spec=spec, batched=True)
        for name, spec in self.actions_spec.items():
            Module.register_tensor(name=name, spec=spec, batched=True)
        Module.register_tensor(name='terminal', spec=dict(type='bool', shape=()), batched=True)
        Module.register_tensor(name='reward', spec=dict(type='float', shape=()), batched=True)
        Module.register_tensor(
            name='deterministic', spec=dict(type='bool', shape=()), batched=False
        )
        Module.register_tensor(name='update', spec=dict(type='bool', shape=()), batched=False)
        Module.register_tensor(name='timestep', spec=dict(type='long', shape=()), batched=False)
        Module.register_tensor(name='episode', spec=dict(type='long', shape=()), batched=False)

    def tf_initialize(self):
        super().tf_initialize()

        self.internals_init = OrderedDict()

        # States
        self.states_input = OrderedDict()
        for name, state_spec in self.states_spec.items():
            self.states_input[name] = self.add_placeholder(
                name=name, dtype=state_spec['type'], shape=state_spec['shape'], batched=True
            )

        # Terminal  (default: False?)
        self.terminal_input = self.add_placeholder(
            name='terminal', dtype='bool', shape=(), batched=True
        )

        # Reward  (default: 0.0?)
        self.reward_input = self.add_placeholder(
            name='reward', dtype='float', shape=(), batched=True
        )

        # Deterministic flag
        self.deterministic_input = self.add_placeholder(
            name='deterministic', dtype='bool', shape=(), batched=False,
            default=tf.constant(value=False, dtype=util.tf_dtype(dtype='bool'))
        )

        # Independent flag
        self.independent_input = self.add_placeholder(
            name='independent', dtype='bool', shape=(), batched=False,
            default=tf.constant(value=False, dtype=util.tf_dtype(dtype='bool'))
        )

        # Parallel index
        self.parallel_input = self.add_placeholder(
            name='parallel', dtype='long', shape=(), batched=False,
            default=tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        )

        # Local timestep
        self.timestep = self.add_variable(
            name='timestep', dtype='long', shape=(self.parallel_interactions,), is_trainable=False,
            initializer='zeros'
        )

        # Local episode
        self.episode = self.add_variable(
            name='episode', dtype='long', shape=(self.parallel_interactions,), is_trainable=False,
            initializer='zeros'
        )

        # States buffer variable
        self.states_buffer = OrderedDict()
        for name, spec in self.states_spec.items():
            self.states_buffer[name] = self.add_variable(
                name=(name + '-buffer'), dtype=spec['type'],
                shape=((self.parallel_interactions, self.buffer_observe) + spec['shape']),
                is_trainable=False
            )

        # Internals buffer variable
        self.internals_buffer = OrderedDict()
        for name, spec in self.internals_spec.items():
            self.internals_buffer[name] = self.add_variable(
                name=(name + '-buffer'), dtype=spec['type'],
                shape=((self.parallel_interactions, self.buffer_observe) + spec['shape']),
                is_trainable=False
            )

        # Actions buffer variable
        self.actions_buffer = OrderedDict()
        for name, spec in self.actions_spec.items():
            self.actions_buffer[name] = self.add_variable(
                name=(name + '-buffer'), dtype=spec['type'],
                shape=((self.parallel_interactions, self.buffer_observe) + spec['shape']),
                is_trainable=False
            )

        # Buffer index
        self.buffer_index = self.add_variable(
            name='buffer-index', dtype='long', shape=(self.parallel_interactions,),
            is_trainable=False, initializer='zeros'
        )

    def tf_regularize(self, states, internals):
        return super().tf_regularize()

    def setup(self):
        """
        Sets up the TensorFlow model graph, starts the servers (distributed mode), creates summarizers
        and savers, initializes (and enters) the TensorFlow session.
        """

        # Create/get our graph, setup local model/global model links, set scope and device.
        graph_default_context = self.setup_graph()

        # Start a tf Server (in case of distributed setup). Only start once.
        # if self.execution_type == "distributed" and self.server is None and self.is_local_model:
        if self.execution_spec is None or self.execution_type == 'local' or not self.is_local_model:
            self.server = None
        else:
            self.start_server()

        # build the graph
        with tf.device(device_name_or_function=self.device):
            # with tf.variable_scope(name_or_scope=self.scope, reuse=False):

            # Create model's "external" components.
            # Create tensorflow functions from "tf_"-methods.

            if self.summarizer_spec is not None:
                with tf.name_scope(name='summarizer'):
                    self.summarizer = tf.contrib.summary.create_file_writer(
                        logdir=self.summarizer_spec['directory'],
                        flush_millis=(self.summarizer_spec.get('flush', 10) * 1000),
                        max_queue=None, filename_suffix=None  # ???
                    )
                    self.summarizer_init = self.summarizer.init()
                    self.summarizer_flush = self.summarizer.flush()
                    self.summarizer_close = self.summarizer.close()
                    default_summarizer = self.summarizer.as_default()
                    default_summarizer.__enter__()
                    # if 'steps' in self.summarizer_spec:
                    #     record_summaries = tf.contrib.summary.record_summaries_every_n_global_steps(
                    #         n=self.summarizer_spec['steps'],
                    #         # global_step=self.global_timestep
                    #     )
                    # else:
                    #     record_summaries = tf.contrib.summary.always_record_summaries()
                    # record_summaries.__enter__()

            self.initialize()

            if self.summary_labels is not None and 'graph' in self.summary_labels:
                with tf.name_scope(name='summarizer'):
                    # summarizer_init = tf.contrib.summary.summary_writer_initializer_op()
                    # assert len(summarizer_init) == 1
                    # initialization = (tf.global_variables_initializer(), summarizer_init[0])
                    graph_def = self.graph.as_graph_def()
                    graph_str = tf.constant(
                        value=graph_def.SerializeToString(), dtype=tf.string, shape=()
                    )
                    self.graph_summary = tf.contrib.summary.graph(
                        param=graph_str, step=self.global_timestep  # episode?
                    )
            else:
                self.graph_summary = None

            if self.summarizer_spec is not None:
                # record_summaries.__exit__(None, None, None)
                default_summarizer.__exit__(None, None, None)

        # If we are a global model -> return here.
        # Saving, syncing, finalizing graph, session is done by local replica model.
        if self.execution_spec is not None and self.execution_type == "distributed" and not self.is_local_model:
            return

        # Saver/Summary -> Scaffold.
        self.setup_saver()

        self.setup_scaffold()

        # Create necessary hooks for the upcoming session.
        hooks = self.setup_hooks()

        # We are done constructing: Finalize our graph, create and enter the session.
        self.setup_session(self.server, hooks, graph_default_context)

    def setup_graph(self):
        """
        Creates our Graph and figures out, which shared/global model to hook up to.
        If we are in a global-model's setup procedure, we do not create
        a new graph (return None as the context). We will instead use the already existing local replica graph
        of the model.

        Returns: None or the graph's as_default()-context.
        """
        graph_default_context = None

        # Single (non-distributed) mode.
        if self.execution_spec is None or self.execution_type == 'single':
            self.graph = tf.Graph()
            graph_default_context = self.graph.as_default()
            graph_default_context.__enter__()
            self.global_model = None

        # Distributed tf
        elif self.execution_type == 'distributed':
            # Parameter-server -> Do not build any graph.
            if self.distributed_spec["job"] == "ps":
                return None

            # worker -> construct the global (main) model; the one hosted on the ps,
            elif self.distributed_spec["job"] == "worker":
                # The local replica model.
                if self.is_local_model:
                    graph = tf.Graph()
                    graph_default_context = graph.as_default()
                    graph_default_context.__enter__()

                    # Now that the graph is created and entered -> deepcopoy ourselves and setup global model first,
                    # then continue.
                    self.global_model = deepcopy(self)
                    # Switch on global construction/setup-mode for the pass to setup().
                    self.global_model.is_local_model = False
                    self.global_model.setup()

                    self.graph = graph
                    self.as_local_model()
                    self.scope += '-worker' + str(self.distributed_spec["task_index"])
                # The global_model (whose Variables are hosted by the ps).
                else:
                    self.graph = tf.get_default_graph()  # lives in the same graph as local model
                    self.global_model = None
                    self.device = tf.train.replica_device_setter(
                        # Place its Variables on the parameter server(s) (round robin).
                        #ps_device="/job:ps",  # default
                        # Train-ops for the global_model are hosted locally (on this worker's node).
                        worker_device=self.device,
                        cluster=self.distributed_spec["cluster_spec"]
                    )
            else:
                raise TensorforceError("Unsupported job type: {}!".format(self.distributed_spec["job"]))
        else:
            raise TensorforceError("Unsupported distributed type: {}!".format(self.distributed_spec["type"]))

        return graph_default_context

    def start_server(self):
        """
        Creates and stores a tf server (and optionally joins it if we are a parameter-server).
        Only relevant, if we are running in distributed mode.
        """
        self.server = tf.train.Server(
            server_or_cluster_def=self.distributed_spec["cluster_spec"],
            job_name=self.distributed_spec["job"],
            task_index=self.distributed_spec["task_index"],
            protocol=self.distributed_spec.get("protocol"),
            config=self.distributed_spec.get("session_config"),
            start=True
        )
        if self.distributed_spec["job"] == "ps":
            self.server.join()
            # This is unreachable?
            quit()

    def setup_saver(self):
        """
        Creates the tf.train.Saver object and stores it in self.saver.
        """
        if self.execution_spec is None or self.execution_type == "single":
            global_variables = self.get_variables()
        else:
            global_variables = self.global_model.get_variables()

        # global_variables += [self.global_episode, self.global_timestep]

        # for c in self.get_savable_components():
        #     c.register_saver_ops()

        # TensorFlow saver object
        # TODO potentially make other options configurable via saver spec.
        self.saver = tf.train.Saver(
            var_list=global_variables,  # should be given?
            reshape=False,
            sharded=False,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=10000.0,
            name=None,
            restore_sequentially=False,
            saver_def=None,
            builder=None,
            defer_build=False,
            allow_empty=False,
            write_version=tf.train.SaverDef.V2,
            pad_step_number=False,
            save_relative_paths=False,
            filename=None
        )

    def setup_scaffold(self):
        """
        Creates the tf.train.Scaffold object and assigns it to self.scaffold.
        Other fields of the Scaffold are generated automatically.
        """
        if self.execution_spec is None or self.execution_type == "single":
            global_variables = self.get_variables()
            # global_variables += [self.global_episode, self.global_timestep]
            init_op = tf.variables_initializer(var_list=global_variables)
            if self.summarizer_spec is not None:
                init_op = tf.group(init_op, self.summarizer_init)
            if self.graph_summary is None:
                ready_op = tf.report_uninitialized_variables(var_list=global_variables)
                ready_for_local_init_op = None
                local_init_op = None
            else:
                ready_op = None
                ready_for_local_init_op = tf.report_uninitialized_variables(var_list=global_variables)
                local_init_op = self.graph_summary

        else:
            # Global and local variable initializers.
            global_variables = self.global_model.get_variables()
            # global_variables += [self.global_episode, self.global_timestep]
            local_variables = self.get_variables()
            init_op = tf.variables_initializer(var_list=global_variables)
            if self.summarizer_spec is not None:
                init_op = tf.group(init_op, self.summarizer_init)
            ready_op = tf.report_uninitialized_variables(var_list=(global_variables + local_variables))
            ready_for_local_init_op = tf.report_uninitialized_variables(var_list=global_variables)
            if self.graph_summary is None:
                local_init_op = tf.group(
                    tf.variables_initializer(var_list=local_variables),
                    # Synchronize values of trainable variables.
                    *(tf.assign(ref=local_var, value=global_var) for local_var, global_var in zip(
                        self.get_variables(only_trainable=True),
                        self.global_model.get_variables(only_trainable=True)
                    ))
                )
            else:
                local_init_op = tf.group(
                    tf.variables_initializer(var_list=local_variables),
                    self.graph_summary,
                    # Synchronize values of trainable variables.
                    *(tf.assign(ref=local_var, value=global_var) for local_var, global_var in zip(
                        self.get_variables(only_trainable=True),
                        self.global_model.get_variables(only_trainable=True)
                    ))
                )

        def init_fn(scaffold, session):
            if self.saver_spec is not None and self.saver_spec.get('load', True):
                directory = self.saver_spec['directory']
                load = self.saver_spec.get('load')
                if isinstance(load, str):
                    save_path = os.path.join(directory, load)
                else:
                    save_path = tf.train.latest_checkpoint(
                        checkpoint_dir=directory, latest_filename=None
                    )
                if save_path is not None:
                    try:
                        scaffold.saver.restore(sess=session, save_path=save_path)
                        session.run(fetches=util.join_scopes(self.name + '.reset', 'timestep-output:0'))
                    except tf.errors.NotFoundError:
                        raise TensorforceError("Error: Existing checkpoint could not be loaded! Set \"load\" to false in saver_spec.")

        # TensorFlow scaffold object
        # TODO explain what it does.
        self.scaffold = tf.train.Scaffold(
            init_op=init_op,
            init_feed_dict=None,
            init_fn=init_fn,
            ready_op=ready_op,
            ready_for_local_init_op=ready_for_local_init_op,
            local_init_op=local_init_op,
            summary_op=None,
            saver=self.saver,
            copy_from_scaffold=None
        )

    def setup_hooks(self):
        """
        Creates and returns a list of hooks to use in a session. Populates self.saver_directory.

        Returns: List of hooks to use in a session.
        """
        hooks = list()

        # Checkpoint saver hook
        if self.saver_spec is not None:  # and (self.execution_type == 'single' or self.distributed_spec['task_index'] == 0):
            self.saver_directory = self.saver_spec['directory']
            self.saver_filename = self.saver_spec.get('filename', 'model')
            hooks.append(tf.train.CheckpointSaverHook(
                checkpoint_dir=self.saver_directory,
                save_secs=self.saver_spec.get('seconds', None if 'steps' in self.saver_spec else 600),
                save_steps=self.saver_spec.get('steps'),  # Either one or the other has to be set.
                saver=None,  # None since given via 'scaffold' argument.
                checkpoint_basename=self.saver_filename, scaffold=self.scaffold, listeners=None
            ))
        else:
            self.saver_directory = None
            self.saver_filename = 'model'

        # Stop at step hook
        # hooks.append(tf.train.StopAtStepHook(
        #     num_steps=???,  # This makes more sense, if load and continue training.
        #     last_step=None  # Either one or the other has to be set.
        # ))

        # # Step counter hook
        # hooks.append(tf.train.StepCounterHook(
        #     every_n_steps=counter_config.get('steps', 100),  # Either one or the other has to be set.
        #     every_n_secs=counter_config.get('secs'),  # Either one or the other has to be set.
        #     output_dir=None,  # None since given via 'summary_writer' argument.
        #     summary_writer=summary_writer
        # ))

        # Other available hooks:
        # tf.train.FinalOpsHook(final_ops, final_ops_feed_dict=None)
        # tf.train.GlobalStepWaiterHook(wait_until_step)
        # tf.train.LoggingTensorHook(tensors, every_n_iter=None, every_n_secs=None)
        # tf.train.NanTensorHook(loss_tensor, fail_on_nan_loss=True)
        # tf.train.ProfilerHook(save_steps=None, save_secs=None, output_dir='', show_dataflow=True, show_memory=False)

        return hooks

    def setup_session(self, server, hooks, graph_default_context):
        """
        Creates and then enters the session for this model (finalizes the graph).

        Args:
            server (tf.train.Server): The tf.train.Server object to connect to (None for single execution).
            hooks (list): A list of (saver, summary, etc..) hooks to be passed to the session.
            graph_default_context: The graph as_default() context that we are currently in.
        """
        if self.execution_spec is not None and self.execution_type == "distributed":
            # if self.distributed_spec['task_index'] == 0:
            # TensorFlow chief session creator object
            session_creator = tf.train.ChiefSessionCreator(
                scaffold=self.scaffold,
                master=server.target,
                config=self.session_config,
                checkpoint_dir=None,
                checkpoint_filename_with_path=None
            )
            # else:
            #     # TensorFlow worker session creator object
            #     session_creator = tf.train.WorkerSessionCreator(
            #         scaffold=self.scaffold,
            #         master=server.target,
            #         config=self.execution_spec.get('session_config'),
            #     )

            # TensorFlow monitored session object
            self.monitored_session = tf.train.MonitoredSession(
                session_creator=session_creator,
                hooks=hooks,
                stop_grace_period_secs=120  # Default value.
            )
            # from tensorflow.python.debug import DumpingDebugWrapperSession
            # self.monitored_session = DumpingDebugWrapperSession(self.monitored_session, self.tf_session_dump_dir)

        else:
            # TensorFlow non-distributed monitored session object
            self.monitored_session = tf.train.SingularMonitoredSession(
                hooks=hooks,
                scaffold=self.scaffold,
                master='',  # Default value.
                config=(None if self.execution_spec is None else self.session_config),  # self.execution_spec.get('session_config'),
                checkpoint_dir=None
            )

        if graph_default_context:
            graph_default_context.__exit__(None, None, None)
        self.graph.finalize()

        # enter the session to be ready for acting/learning
        self.monitored_session.__enter__()
        self.session = self.monitored_session._tf_sess()

    def close(self):
        """
        Saves the model (of saver dir is given) and closes the session.
        """
        if self.summarizer_spec is not None:
            self.monitored_session.run(fetches=self.summarizer_close)
        if self.saver_spec is not None:
            self.save(append_timestep=True)
        self.monitored_session.__exit__(None, None, None)

    def as_local_model(self):
        pass

    def tf_core_act(self, states, internals):
        """
        Creates and returns the TensorFlow operations for retrieving the actions and - if applicable -
        the posterior internal state Tensors in reaction to the given input states (and prior internal states).

        Args:
            states (dict): Dict of state tensors (each key represents one state space component).
            internals (dict): Dict of internal state tensors (each key represents one internal space component).

        Returns:
            tuple:
                1) dict of output actions (with or without exploration applied (see `deterministic`))
                2) list of posterior internal state Tensors (empty for non-internal state models)
        """
        raise NotImplementedError

    def tf_core_observe(self, states, internals, actions, terminal, reward):
        """
        Creates the TensorFlow operations for processing a batch of observations coming in from our buffer (state,
        action, internals) as well as from the agent's python-batch (terminal-signals and rewards from the env).

        Args:
            states (dict): Dict of state tensors (each key represents one state space component).
            internals (dict): Dict of prior internal state tensors (each key represents one internal state component).
            actions (dict): Dict of action tensors (each key represents one action space component).
            terminal: 1D (bool) tensor of terminal signals.
            reward: 1D (float) tensor of rewards.

        Returns:
            The observation operation depending on the model type.
        """
        raise NotImplementedError

    def tf_preprocess(self, states, actions, reward):
        """
        Applies preprocessing ops to the raw states/action/reward inputs.

        Args:
            states (dict): Dict of raw state tensors.
            actions (dict): Dict or raw action tensors.
            reward: 1D (float) raw rewards tensor.

        Returns: The preprocessed versions of the input tensors.
        """
        # States preprocessing
        for name in sorted(self.states_preprocessing):
            states[name] = self.states_preprocessing[name].process(tensor=states[name])

        # Reward preprocessing
        if self.reward_preprocessing is not None:
            reward = self.reward_preprocessing.process(tensor=reward)

        return states, actions, reward

    def api_reset(self):
        assignment = self.buffer_index.assign(
            value=tf.zeros(shape=(self.parallel_interactions,), dtype=util.tf_dtype(dtype='long')),
            read_value=False
        )

        # Synchronization initial sync?

        with tf.control_dependencies(control_inputs=(assignment,)):
            timestep = util.identity_operation(
                x=self.global_timestep, operation_name='timestep-output'
            )
            episode = util.identity_operation(
                x=self.global_episode, operation_name='episode-output'
            )

        return timestep, episode

    def api_act(self):
        """
        Creates and stores tf operations that are fetched when calling act(): actions_output, internals_output and
        timestep_output.

        Args:
            states (dict): Dict of state tensors (each key represents one state space component).
            deterministic: 0D (bool) tensor (whether to not use action exploration).
            independent (bool): 0D (bool) tensor (whether to store states/internals/action in local buffer).
            parallel (bool): ???
        """

        # Inputs
        states = self.states_input
        parallel = self.parallel_input
        deterministic = self.deterministic_input
        independent = self.independent_input

        # Assertions
        assertions = list()
        # states: type and shape
        for name, spec in self.states_spec.items():
            assertions.append(
                tf.debugging.assert_type(
                    tensor=states[name], tf_type=util.tf_dtype(dtype=spec['type'])
                )
            )
            # assertions.append(
            #     tf.debugging.assert_equal(
            #         x=tf.shape(input=states[name], out_type=util.tf_dtype(dtype='int')),
            #         y=tf.constant(value=spec['shape'], dtype=util.tf_dtype(dtype='int'))
            #     )
            # )
        # parallel: type, shape and value
        assertions.append(
            tf.debugging.assert_type(tensor=parallel, tf_type=util.tf_dtype(dtype='long'))
        )
        assertions.append(tf.debugging.assert_scalar(tensor=parallel))
        assertions.append(tf.debugging.assert_non_negative(x=parallel))
        assertions.append(
            tf.debugging.assert_less(
                x=parallel,
                y=tf.constant(value=self.parallel_interactions, dtype=util.tf_dtype(dtype='long'))
            )
        )
        # deterministic: type and shape
        assertions.append(
            tf.debugging.assert_type(tensor=deterministic, tf_type=util.tf_dtype(dtype='bool'))
        )
        assertions.append(tf.debugging.assert_scalar(tensor=deterministic))
        # independent: type and shape
        assertions.append(
            tf.debugging.assert_type(tensor=independent, tf_type=util.tf_dtype(dtype='bool'))
        )
        assertions.append(tf.debugging.assert_scalar(tensor=independent))

        # Set global tensors
        Module.update_tensors(
            deterministic=deterministic, update=tf.constant(value=False),
            timestep=self.timestep[parallel], episode=self.episode[parallel]
        )

        # Increment timestep
        with tf.control_dependencies(control_inputs=assertions):

            def increment_timestep():
                assignments = list()
                one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
                assignments.append(
                    self.timestep.scatter_nd_add(indices=[(parallel,)], updates=[one])
                )
                assignments.append(self.global_timestep.assign_add(delta=one, read_value=False))
                with tf.control_dependencies(control_inputs=assignments):
                    return util.no_operation()

            incremented_timestep = self.cond(
                pred=independent, true_fn=util.no_operation, false_fn=increment_timestep
            )

        # Variable noise
        variables = self.get_variables(only_trainable=True)
        if len(variables) > 0:
            with tf.control_dependencies(control_inputs=(incremented_timestep,)):
                variable_noise = self.variable_noise.value()

                def no_variable_noise():
                    noise_tensors = [
                        tf.zeros_like(tensor=variable, dtype=util.tf_dtype('float'))
                        for variable in variables
                    ]
                    return noise_tensors

                def apply_variable_noise():
                    assignments = list()
                    noise_tensors = list()
                    for variable in variables:
                        noise = tf.random.normal(
                            shape=util.shape(variable), mean=0.0, stddev=variable_noise,
                            dtype=util.tf_dtype('float')
                        )
                        noise_tensors.append(noise)
                        assignments.append(variable.assign_add(delta=noise, read_value=False))
                    with tf.control_dependencies(control_inputs=assignments):
                        return [util.identity_operation(x=noise) for noise in noise_tensors]

                zero = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))
                skip_variable_noise = tf.logical_or(
                    x=deterministic, y=tf.math.equal(x=variable_noise, y=zero)
                )
                variable_noise_tensors = self.cond(
                    pred=skip_variable_noise, true_fn=no_variable_noise,
                    false_fn=apply_variable_noise
                )

        else:
            variable_noise_tensors = (incremented_timestep,)

        # Initialize or retrieve internals
        if len(self.internals_spec) > 0:
            with tf.control_dependencies(control_inputs=variable_noise_tensors):
                buffer_index = self.buffer_index[parallel]
                one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))

                def initialize_internals():
                    internals = OrderedDict()
                    for name, init in self.internals_init.items():
                        internals[name] = tf.expand_dims(input=init, axis=0)
                    return internals

                def retrieve_internals():
                    internals = OrderedDict()
                    for name in self.internals_spec:
                        internals[name] = tf.gather_nd(
                            params=self.internals_buffer[name],
                            indices=[(parallel, buffer_index - one)]
                        )
                    return internals

                zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
                initialize = tf.math.equal(x=buffer_index, y=zero)
                internals = self.cond(
                    pred=initialize, true_fn=initialize_internals, false_fn=retrieve_internals
                )
                retrieved_internals = util.flatten(xs=internals)

        else:
            internals = OrderedDict()
            retrieved_internals = variable_noise_tensors

        # Core act: retrieve actions and internals
        with tf.control_dependencies(control_inputs=retrieved_internals):
            Module.update_tensors(**states, **internals)
            actions, internals = self.core_act(states=states, internals=internals)
            Module.update_tensors(**actions)

        # Exploration
        with tf.control_dependencies(
            control_inputs=(util.flatten(xs=actions) + util.flatten(xs=internals))
        ):
            if not isinstance(self.exploration, dict):
                exploration = self.exploration.value()

            for name, spec in self.actions_spec.items():
                if isinstance(self.exploration, dict):
                    if name in self.exploration:
                        exploration = self.exploration[name].value()
                    else:
                        continue

                def no_exploration():
                    return actions[name]

                if spec['type'] == 'bool':
                    float_dtype = util.tf_dtype(dtype='float')

                    def apply_exploration():
                        shape = tf.shape(input=actions[name])
                        condition = tf.random.uniform(shape=shape, dtype=float_dtype) < exploration
                        half = tf.constant(value=0.5, dtype=float_dtype)
                        random_action = tf.random.uniform(shape=shape, dtype=float_dtype) < half
                        return tf.where(condition=condition, x=random_action, y=actions[name])

                elif spec['type'] == 'int':
                    float_dtype = util.tf_dtype(dtype='float')

                    def apply_exploration():
                        shape = tf.shape(input=actions[name])
                        condition = tf.random.uniform(shape=shape, dtype=float_dtype) < exploration
                        random_action = tf.random_uniform(
                            shape=shape, maxval=spec['num_values'], dtype=float_dtype
                        )
                        random_action = tf.dtypes.cast(x=random_action, dtype=util.tf_dtype('int'))
                        return tf.where(condition=condition, x=random_action, y=actions[name])

                elif spec['type'] == 'float':
                    if 'min_value' in spec:

                        def apply_exploration():
                            return tf.clip_by_value(
                                t=(actions[name] + exploration), clip_value_min=spec['min_value'],
                                clip_value_max=spec['max_value']
                            )

                    else:

                        def apply_exploration():
                            return actions[name] + exploration

                zero = tf.constant(value=0.0, dtype=util.tf_dtype('float'))
                skip_exploration = tf.math.logical_or(
                    x=deterministic, y=tf.math.equal(x=exploration, y=zero)
                )
                actions[name] = self.cond(
                    pred=skip_exploration, true_fn=no_exploration, false_fn=apply_exploration
                )

        # Variable noise
        if len(variables) > 0:
            with tf.control_dependencies(control_inputs=util.flatten(xs=actions)):

                def reverse_variable_noise():
                    assignments = list()
                    for variable, noise in zip(variables, variable_noise_tensors):
                        assignments.append(variable.assign_sub(delta=noise, read_value=False))
                    with tf.control_dependencies(control_inputs=assignments):
                        return util.no_operation()

                reversed_variable_noise = self.cond(
                    pred=skip_variable_noise, true_fn=util.no_operation,
                    false_fn=reverse_variable_noise
                )
                reversed_variable_noise = (reversed_variable_noise,)

        else:
            reversed_variable_noise = util.flatten(xs=actions)

        # Update states/internals/actions buffers
        with tf.control_dependencies(control_inputs=reversed_variable_noise):

            def update_buffers():
                operations = list()
                buffer_index = self.buffer_index[parallel]
                for name in self.states_spec:
                    operations.append(
                        self.states_buffer[name].scatter_nd_update(
                            indices=[(parallel, buffer_index)], updates=states[name]
                        )
                    )
                for name in self.internals_spec:
                    operations.append(
                        self.internals_buffer[name].scatter_nd_update(
                            indices=[(parallel, buffer_index)], updates=internals[name]
                        )
                    )
                for name in self.actions_spec:
                    operations.append(
                        self.actions_buffer[name].scatter_nd_update(
                            indices=[(parallel, buffer_index)], updates=actions[name]
                        )
                    )

                # Increment buffer index
                with tf.control_dependencies(control_inputs=operations):
                    one = tf.constant(value=1, dtype=util.tf_dtype(dtype='int'))
                    incremented_buffer_index = self.buffer_index.scatter_nd_add(
                        indices=[(parallel,)], updates=[one]
                    )

                with tf.control_dependencies(control_inputs=(incremented_buffer_index,)):
                    return util.no_operation()

            updated_buffers = self.cond(
                pred=independent, true_fn=util.no_operation, false_fn=update_buffers
            )

        # Return timestep
        with tf.control_dependencies(control_inputs=(updated_buffers,)):
            # Function-level identity operation for retrieval (plus enforce dependency)
            for name, spec in self.actions_spec.items():
                actions[name] = util.identity_operation(
                    x=actions[name], operation_name=(name + '-output')
                )
            timestep = util.identity_operation(
                x=self.global_timestep, operation_name='timestep-output'
            )

        return actions, timestep

    def api_observe(self):
        """
        Returns the tf op to fetch when an observation batch is passed in (e.g. an episode's rewards and
        terminals). Uses the filled tf buffers for states, actions and internals to run
        the tf_observe_timestep (model-dependent), resets buffer index and increases counters (episodes,
        timesteps).

        Args:
            terminal: The 1D tensor (bool) of terminal signals to process (more than one True within that list is ok).
            reward: The 1D tensor (float) of rewards to process.

        Returns: Tf op to fetch when `observe()` is called.
        """

        # Inputs
        terminal = self.terminal_input
        reward = self.reward_input
        parallel = self.parallel_input

        # Assertions
        assertions = list()
        # terminal: type and shape
        assertions.append(
            tf.debugging.assert_type(tensor=terminal, tf_type=util.tf_dtype(dtype='bool'))
        )
        assertions.append(tf.debugging.assert_rank(x=terminal, rank=1))
        # reward: type and shape
        assertions.append(
            tf.debugging.assert_type(tensor=reward, tf_type=util.tf_dtype(dtype='float'))
        )
        assertions.append(tf.debugging.assert_rank(x=reward, rank=1))
        # parallel: type, shape and value
        assertions.append(
            tf.debugging.assert_type(tensor=parallel, tf_type=util.tf_dtype(dtype='long'))
        )
        assertions.append(tf.debugging.assert_scalar(tensor=parallel))
        assertions.append(tf.debugging.assert_non_negative(x=parallel))
        assertions.append(
            tf.debugging.assert_less(
                x=parallel,
                y=tf.constant(value=self.parallel_interactions, dtype=util.tf_dtype(dtype='long'))
            )
        )
        # shape of terminal equals shape of reward
        assertions.append(
            tf.debugging.assert_equal(x=tf.shape(input=terminal), y=tf.shape(input=reward))
        )
        # size of terminal equals buffer index
        assertions.append(
            tf.debugging.assert_equal(
                x=tf.shape(input=terminal)[0],
                y=tf.dtypes.cast(x=self.buffer_index[parallel], dtype=tf.int32)
            )
        )
        # at most one terminal
        assertions.append(
            tf.debugging.assert_less_equal(
                x=tf.math.count_nonzero(input_tensor=terminal, dtype=util.tf_dtype(dtype='int')),
                y=tf.constant(value=1, dtype=util.tf_dtype(dtype='int'))
            )
        )
        # if terminal, last timestep in batch
        assertions.append(
            tf.debugging.assert_equal(x=tf.math.reduce_any(input_tensor=terminal), y=terminal[-1])
        )

        # Set global tensors
        Module.update_tensors(
            deterministic=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool')),
            update=tf.constant(value=False, dtype=util.tf_dtype(dtype='bool')),
            timestep=self.timestep[parallel], episode=self.episode[parallel]
        )

        # Increment episode
        def increment_episode():
            assignments = list()
            one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
            assignments.append(self.episode.scatter_nd_add(indices=[(parallel,)], updates=[one]))
            assignments.append(self.global_episode.assign_add(delta=one, read_value=False))
            with tf.control_dependencies(control_inputs=assignments):
                return util.no_operation()

        with tf.control_dependencies(control_inputs=assertions):
            episode_finished = tf.reduce_any(input_tensor=terminal, axis=0)
            incremented_episode = self.cond(
                pred=episode_finished, true_fn=increment_episode, false_fn=util.no_operation
            )

        # Core observe: retrieve observation operation
        with tf.control_dependencies(control_inputs=(incremented_episode,)):
            buffer_index = self.buffer_index[parallel]
            states = OrderedDict()
            for name in self.states_spec:
                states[name] = self.states_buffer[name][parallel, :buffer_index]
            internals = OrderedDict()
            for name in self.internals_spec:
                internals[name] = self.internals_buffer[name][parallel, :buffer_index]
            actions = OrderedDict()
            for name in self.actions_spec:
                actions[name] = self.actions_buffer[name][parallel, :buffer_index]

            Module.update_tensors(
                **states, **internals, **actions, terminal=terminal, reward=reward
            )
            reward = self.add_summary(label='reward', name='reward', tensor=reward)
            observation = self.core_observe(
                states=states, internals=internals, actions=actions, terminal=terminal,
                reward=reward
            )

        # Reset buffer index
        with tf.control_dependencies(control_inputs=(observation,)):
            zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='int'))

            reset_buffer_index = self.buffer_index.scatter_nd_update(
                indices=[(parallel,)], updates=[zero]
            )

        # Return episode
        with tf.control_dependencies(control_inputs=(reset_buffer_index,)):
            # Function-level identity operation for retrieval (plus enforce dependency)
            episode = util.identity_operation(
                x=self.global_episode, operation_name='episode-output'
            )

        return episode

        # TODO: add up rewards per episode and add summary_label 'episode-reward'

    def save(self, directory=None, filename=None, append_timestep=True):
        """
        Save TensorFlow model. If no checkpoint directory is given, the model's default saver
        directory is used. Optionally appends current timestep to prevent overwriting previous
        checkpoint files. Turn off to be able to load model from the same given path argument as
        given here.

        Args:
            directory: Optional checkpoint directory.
            append_timestep: Appends the current timestep to the checkpoint file if true.

        Returns:
            Checkpoint path where the model was saved.
        """
        if self.summarizer_spec is not None:
            self.monitored_session.run(fetches=self.summarizer_flush)

        if directory is None:
            assert self.saver_directory
            directory = self.saver_directory
        if filename is None:
            filename = self.saver_filename
        save_path = os.path.join(directory, filename)

        return self.saver.save(
            sess=self.session, save_path=save_path,
            global_step=(self.global_timestep if append_timestep else None),
            # latest_filename=None,  # Defaults to 'checkpoint'.
            meta_graph_suffix='meta', write_meta_graph=True, write_state=True
        )

    def restore(self, directory=None, filename=None):
        """
        Restore TensorFlow model. If no checkpoint file is given, the latest checkpoint is
        restored. If no checkpoint directory is given, the model's default saver directory is
        used (unless file specifies the entire path).

        Args:
            directory: Optional checkpoint directory.
            file: Optional checkpoint file, or path if directory not given.
        """
        if directory is None:
            assert self.saver_directory
            directory = self.saver_directory
        if filename is None:
            save_path = tf.train.latest_checkpoint(checkpoint_dir=directory, latest_filename=None)
        else:
            save_path = os.path.join(directory, filename)

        self.saver.restore(sess=self.session, save_path=save_path)
        return self.reset()

    # def save_component(self, component_name, save_path):
    #     """
    #     Saves a component of this model to the designated location.

    #     Args:
    #         component_name: The component to save.
    #         save_path: The location to save to.
    #     Returns:
    #         Checkpoint path where the component was saved.
    #     """
    #     component = self.get_component(component_name=component_name)
    #     self._validate_savable(component=component, component_name=component_name)
    #     return component.save(sess=self.session, save_path=save_path)

    # def restore_component(self, component_name, save_path):
    #     """
    #     Restores a component's parameters from a save location.

    #     Args:
    #         component_name: The component to restore.
    #         save_path: The save location.
    #     """
    #     component = self.get_component(component_name=component_name)
    #     self._validate_savable(component=component, component_name=component_name)
    #     component.restore(sess=self.session, save_path=save_path)
