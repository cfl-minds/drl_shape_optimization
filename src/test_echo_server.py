from __future__ import print_function
from echo_server import EchoServer
import numpy as np

# check with a list
print()
print("Test with a list")
data = [1, 2, 3]

echo_server_instance = EchoServer(verbose=5)
encoded = echo_server_instance.encode_message('DUMP', data)

print(encoded)

decoded = echo_server_instance.decode_message(encoded)

print(decoded)

# check with a numpy array
print()
print("Test with a np array")
data = np.array([1.0, 2.0, 3.0])

echo_server_instance = EchoServer(verbose=5)
encoded = echo_server_instance.encode_message('DUMP', data)

print(encoded)

decoded = echo_server_instance.decode_message(encoded)

print(decoded)

# check with a dict
print()
print("Test with a np array")
data = {'a': np.array([1.0, 2.0, 3.0]), 'b': [1, 2, 3], 4 : "blabla"}

echo_server_instance = EchoServer(verbose=5)
encoded = echo_server_instance.encode_message('DUMP', data)

print(encoded)

decoded = echo_server_instance.decode_message(encoded)

print(decoded)
