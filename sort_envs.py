# Generic imports
import os, sys, glob, shutil
import numpy as np

# Find the number of envs
main_dir = '.'
envs = [f.path for f in os.scandir(main_dir) if f.is_dir()]

# Process names
tmp = []
for env in envs:
    env = env[2:]
    if (env[0:3] == 'env'):
        tmp.append(env)

# Printing
envs = tmp
print('I found ',str(len(envs)),' environments')

# Create final dirs if necessary
path      = 'sorted_envs'
png_path  = path+'/png'
csv_path  = path+'/csv'
sol_path  = path+'/sol'
best_path = path+'/best'
if (not os.path.isdir(path)):
    os.mkdir(path)
if (not os.path.isdir(png_path)):
    os.mkdir(png_path)
if (not os.path.isdir(csv_path)):
    os.mkdir(csv_path)
if (not os.path.isdir(sol_path)):
    os.mkdir(sol_path)
if (not os.path.isdir(best_path)):
    os.mkdir(best_path)

# Read env contents
n_outputs   = 10
looping     = True
reward      = []
glb_index   = 1
loc_index   = 0
avg_reward  = []
avg_rew     = 0.0

# Loop until no more shapes can be found
while looping:
    # Copy loc index to check if loop must be stopped
    loc_index_cp = loc_index

    # Loop over envs
    for env in envs:
        img    = env+'/save/png/shape_'+str(glb_index)+'.png'
        csv    = env+'/save/csv/shape_'+str(glb_index)+'.csv'
        sol    = env+'/save/sol/'+str(glb_index)+'.png'
        sol_u  = env+'/save/sol/'+str(glb_index)+'_u.png'
        sol_v  = env+'/save/sol/'+str(glb_index)+'_v.png'
        sol_p  = env+'/save/sol/'+str(glb_index)+'_p.png'

        # If files exists, copy
        if os.path.isfile(img):
            shutil.copy(img,   png_path+'/'+str(loc_index)+'.png')
        if os.path.isfile(csv):
            shutil.copy(csv,   csv_path+'/'+str(loc_index)+'.csv')
        if os.path.isfile(sol_u):
            shutil.copy(sol_u, sol_path+'/'+str(loc_index)+'_u.png')
        if os.path.isfile(sol_v):
            shutil.copy(sol_v, sol_path+'/'+str(loc_index)+'_v.png')
        if os.path.isfile(sol_p):
            shutil.copy(sol_p, sol_path+'/'+str(loc_index)+'_p.png')
        if os.path.isfile(sol):
            shutil.copy(sol,   sol_path+'/'+str(loc_index)+'.png')

            # All the following is done only if computation ended well
            # Store reward and check max reward
            filename = env+'/save/reward_penalization'
            line     = None

            with open(filename) as f:
                line = f.read().split('\n')[glb_index-1]
                line = line.split(' ')

            if (len(line)>1):
                rew         = float(line[1])
                avg_rew     += rew
                avg_reward.append(avg_rew/(loc_index+1))
                reward.append(rew)

            # Update index
            loc_index += 1

    # Stop looping if index has not changed
    if (loc_index == loc_index_cp):
        looping = False

    # Update global index
    glb_index += 1

# Sort reward
sort_rew = np.argsort(-1.0*np.asarray(reward))

# Write reward to file
filename = path+'/reward'
with open(filename, 'w') as f:
    for i in range(len(reward)):
        f.write(str(i)+' ')
        f.write(str(reward[i])+' ')
        f.write(str(avg_reward[i]))
        f.write('\n')

# Copy best solutions
for i in range(n_outputs):
    img = png_path+'/'+str(sort_rew[i])+'.png'
    if os.path.isfile(img):
        shutil.copy(img,   best_path+'/.')

# Printing
print('I found '+str(loc_index)+' shapes in total')
print('Best rewards are:')
for i in range(n_outputs):
    print('                  '+str(reward[sort_rew[i]])+' for shape '+str(sort_rew[i]))
