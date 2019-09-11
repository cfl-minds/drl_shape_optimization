from multiprocessing import Process
import time
import argparse
import socket
import os
from utils import check_ports_avail

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
ap.add_argument("-p", "--ports-start", required=True, help="the start of the range of ports to use", type=int)
ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)

args = vars(ap.parse_args())

number_servers = args["number_servers"]
ports_start = args["ports_start"]
host = args["host"]

if host == 'None':
    host = socket.gethostname()

list_ports = [ind_server + ports_start for ind_server in range(number_servers)]

# check for the availability of the ports
if not check_ports_avail(host, list_ports):
    quit()

# make copies of the code to avoid collisions
for rank in range(number_servers):

    print("copy files for env of rank {}".format(rank))

    # the necessary files and folders
    list_to_copy = ['environment.py',
                    'fenics_solver.py',
                    'generate_shape.py',
                    'meshes_utils.py',
                    'shapes_utils.py',
                    'parametered_env.py',
                    'start_one_server.py',
                    'reset']

    # make the env folder and copy all the necessary files
    if not os.path.exists('env_' + str(rank)):
        os.system('mkdir ' + 'env_' + str(rank) + '/')

    for crrt_to_copy in list_to_copy:
        os.system('cp -r ' + crrt_to_copy + ' env_' + str(rank) + '/.')

def launch_one_server(rank, host, port):
    os.system('cd env_{} && python3 start_one_server.py -t {} -p {}'.format(rank, host, port))	        

processes = []

# launch all the servers one after the other
for rank, port in enumerate(list_ports):
    print("launching process of rank {}".format(rank))
    proc = Process(target=launch_one_server, args=(rank, host, port))
    proc.start()
    processes.append(proc)
    time.sleep(2.0)  # just to avoid collisions in the terminal printing

print("all processes started, ready to serve...")

for proc in processes:
    proc.join()
