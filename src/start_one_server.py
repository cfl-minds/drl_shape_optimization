import os
import argparse
import socket
import RemoteEnvironmentServer
from   RemoteEnvironmentServer import RemoteEnvironmentServer
from   parametered_env         import *

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--port", required=True, help="the port to use", type=int)
ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)

args = vars(ap.parse_args())

port = args["port"]
host = args["host"]

if host == 'None':
    host = socket.gethostname()

cwd = os.getcwd()

print("from {} lauching one server".format(cwd))

tensorforce_environment = resume_env()
RemoteEnvironmentServer(tensorforce_environment, host=host, port=port)
