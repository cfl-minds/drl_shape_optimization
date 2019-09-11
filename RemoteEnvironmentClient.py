from threading import Thread
from tensorforce import TensorforceError
from tensorforce.environments import Environment
import socket
from echo_server import EchoServer


class RemoteEnvironmentClient(Environment):
    """Used to communicate with a RemoteEnvironmentServer. The idea is that the pair
    (RemoteEnvironmentClient, RemoteEnvironmentServer) allows to transmit information
    through a socket seamlessly.

    The RemoteEnvironmentClient can be directly given to the Runner.

    The RemoteEnvironmentServer herits from a valid Environment add adds the socketing.
    """

    def __init__(self,
                 example_environment,
                 port=12230,
                 host=None,
                 verbose=1,
                 buffer_size=262144
                 ):
        """(port, host) is the necessary info for connecting to the Server socket.
        """

        # templated tensorforce stuff
        self.observation = None
        self.thread = None

        self.buffer_size = buffer_size

        # make arguments available to the class
        # socket
        self.port = port
        self.host = host
        # misc
        self.verbose = verbose
        # states and actions
        self.example_environment = example_environment
        self.dict_states = example_environment.states
        self.dict_actions = example_environment.actions

        # start the socket
        self.valid_socket = False
        self.socket = socket.socket()
        # if necessary, use the local host
        if self.host is None:
            self.host = socket.gethostname()
        # connect to the socket
        self.socket.connect((self.host, self.port))
        if self.verbose > 0:
            print('Connected to {}:{}'.format(self.host, self.port))
        # now the socket is ok
        self.valid_socket = True

        self.episode = 0
        self.step = 0

    def states(self):
        """
        Return the state space. Might include subdicts if multiple states are
        available simultaneously.

        Returns:
            States specification, with the following attributes
                (required):
                - type: one of 'bool', 'int', 'float' (default: 'float').
                - shape: integer, or list/tuple of integers (required).
        """

        return(self.dict_states)

    def actions(self):
        """
        Return the action space. Might include subdicts if multiple actions are
        available simultaneously.

        Returns:
            actions (spec, or dict of specs): Actions specification, with the following attributes
                (required):
                - type: one of 'bool', 'int', 'float' (required).
                - shape: integer, or list/tuple of integers (default: []).
                - num_actions: integer (required if type == 'int').
                - min_value and max_value: float (optional if type == 'float', default: none).
        """

        return(self.dict_actions)

    def seed(self, seed):
        """
        Sets the random seed of the environment to the given value (current time, if seed=None).
        Naturally deterministic Environments (e.g. ALE or some gym Envs) don't have to implement this method.

        Args:
            seed (int): The seed to use for initializing the pseudo-random number generator (default=epoch time in sec).
        Returns: The actual seed (int) used OR None if Environment did not override this method (no seeding supported).
        """

        # TODO: implement it through the socket
        return None

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """

        # TODO: think about sending a killing message to the server? Maybe not necessary - can reuse the
        # server maybe - but may be needed if want to clean after us.

        if self.valid_socket:
            self.socket.close()


    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """

        # perform the reset
        _ = self.communicate_socket("RESET", 1)

        # get the state
        _, init_state = self.communicate_socket("STATE", 1)

        # Updating episode and step numbers
        self.episode += 1
        self.step = 0

        if self.verbose > 1:
            print("reset done; init_state:")
            print(init_state)

        return(init_state)

    def execute(self, actions):
        """
        Executes action, observes next state(s) and reward.

        Args:
            actions: Actions to execute.

        Returns:
            Tuple of (next state, bool indicating terminal, reward)
        """

        # send the control message
        self.communicate_socket("CONTROL", actions)

        # ask to evolve
        self.communicate_socket("EVOLVE", 1)

        # obtain the next state
        _, next_state = self.communicate_socket("STATE", 1)

        # check if terminal
        _, terminal = self.communicate_socket("TERMINAL", 1)

        # get the reward
        _, reward = self.communicate_socket("REWARD", 1)

        # now we have done one more step
        self.step += 1

        if self.verbose > 1:
            print("execute performed; state, terminal, reward:")
            print(next_state)
            print(terminal)
            print(reward)

        return (next_state, terminal, reward)

    def communicate_socket(self, request, data):
        """Send a request through the socket, and wait for the answer message.
        """

        to_send = EchoServer.encode_message(request, data, verbose=self.verbose)
        self.socket.send(to_send)

        # TODO: the recv argument gives the max size of the buffer, can be a source of missouts if
        # a message is larger than this; add some checks to verify that no overflow
        received_msg = self.socket.recv(self.buffer_size)

        request, data = EchoServer.decode_message(received_msg, verbose=self.verbose)

        return(request, data)
