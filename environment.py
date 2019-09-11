# Generic imports
import os
import glob
import math
import time
import PIL
import matplotlib
import numpy             as np
import matplotlib.pyplot as plt

# Custom imports
from shapes_utils  import *
from meshes_utils  import *
from fenics_solver import *

# Define environment class for rl
class env():

    # Static variable
    episode_nb =-1
    control_nb = 0

    # Initialize empty shape
    shape = Shape()

    def __init__(self,
                 nb_pts_to_move, pts_to_move,
                 nb_ctrls_per_episode, nb_episodes,
                 max_deformation,
                 restart_from_cylinder,
                 replace_shape,
                 comp_dir,
                 restore_model,
                 saving_model_period,
                 final_time, cfl, reynolds,
                 output,
                 shape_h, domain_h,
                 cell_limit,
                 reset_dir,
                 xmin, xmax, ymin, ymax):

        self.nb_pts_to_move          = nb_pts_to_move
        self.pts_to_move             = pts_to_move
        self.nb_ctrls_per_episode    = nb_ctrls_per_episode
        self.nb_episodes             = nb_episodes
        self.max_deformation         = max_deformation
        self.restart_from_cylinder   = restart_from_cylinder
        self.replace_shape           = replace_shape
        self.comp_dir                = comp_dir
        self.restore_model           = restore_model
        self.final_time              = final_time
        self.cfl                     = cfl
        self.reynolds                = reynolds
        self.output                  = output
        self.shape_h                 = shape_h
        self.domain_h                = domain_h
        self.cell_limit              = cell_limit
        self.reset_dir               = reset_dir
        self.xmin                    = xmin
        self.xmax                    = xmax
        self.ymin                    = ymin
        self.ymax                    = ymax

        # Saving model periodically
        env.saving_model_period = saving_model_period

        # Check that reset dir exists
        if (not os.path.exists('./'+self.reset_dir)):
            print('Error : I could not find the reset folder')
            exit()

        # Initialize shape by reading it from reset folder
        # Shape reset is automatic when reading from csv
        env.shape.read_csv(self.reset_dir+'/shape_0.csv')
        env.shape.generate(centering=False)

        # Initialize arrays
        self.drag       = np.array([])
        self.lift       = np.array([])
        self.reward     = np.array([])
        self.avg_drag   = np.array([])
        self.avg_lift   = np.array([])
        self.avg_reward = np.array([])
        self.penal      = np.array([])

        # If restore model, get last increment
        if (self.restore_model):
            file_lst        = glob.glob(self.comp_dir+'/save/png/*.png')
            last_file       = max(file_lst, key=os.path.getctime)
            tmp             = last_file.split('_')[-1]
            env.shape.index = int(tmp.split('.')[0])
            print('Restarting from shape index '+str(env.shape.index))

        # Remove save folder
        if (not self.restore_model):
            if (os.path.exists(self.comp_dir+'/save')):
                os.system('rm -r '+self.comp_dir+'/save')

            # Make sure the save repo exists and is properly formated
            if (not os.path.exists(self.comp_dir+'/save')):
                os.system('mkdir '+self.comp_dir+'/save')
            if (not os.path.exists(self.comp_dir+'/save/png')):
                os.system('mkdir '+self.comp_dir+'/save/png')
            if (not os.path.exists(self.comp_dir+'/save/rejected')):
                os.system('mkdir '+self.comp_dir+'/save/rejected')
            if (not os.path.exists(self.comp_dir+'/save/xml')):
                os.system('mkdir '+self.comp_dir+'/save/xml')
            if (not os.path.exists(self.comp_dir+'/save/csv')):
                os.system('mkdir '+self.comp_dir+'/save/csv')
            if (not os.path.exists(self.comp_dir+'/save/sol')):
                os.system('mkdir '+self.comp_dir+'/save/sol')

            # Copy initial files in save repo if restart from cylinder
            if (self.restart_from_cylinder):
                os.system('cp '+self.reset_dir+'/shape_0.png '+self.comp_dir+'/save/png/.')
                os.system('cp '+self.reset_dir+'/shape_0.xml '+self.comp_dir+'/save/xml/.')
                os.system('cp '+self.reset_dir+'/shape_0.csv '+self.comp_dir+'/save/csv/.')

    def reset(self):
        # Console output
        env.episode_nb += 1
        print('****** Starting episode '+str(env.episode_nb))
        if (env.episode_nb%100 == 0): time.sleep(10)

        # Reset control number
        env.control_nb  = 0

        # Reset from cylinder if asked
        if (self.restart_from_cylinder):
            env.shape.read_csv(self.reset_dir+'/shape_0.csv', keep_numbering=True)
            env.shape.generate(centering=False)

        # Fill next state
        next_state = self.fill_next_state(True, 0)

        return(next_state)

    def execute(self, action=None):
        # Console output
        print('***    Starting control '+str(env.control_nb))

        # Convert actions to numpy array
        deformation = np.array(action).reshape((int(len(action)/3), 3))

        for i in range(self.nb_pts_to_move):
            pt     = self.pts_to_move[i]
            radius = max(abs(deformation[i,0]),0.2)*self.max_deformation
            dangle = (360.0/float(env.shape.n_control_pts))
            angle  = dangle*float(pt)+deformation[i,1]*dangle/2.0
            x      = radius*math.cos(math.radians(angle))
            y      = radius*math.sin(math.radians(angle))
            edg    = 0.5+0.5*abs(deformation[i,2])

            deformation[i,0] = x
            deformation[i,1] = y
            deformation[i,2] = edg

        # Modify shape
        env.shape.modify_shape_from_field(deformation,
                                          replace=self.replace_shape,
                                          pts_list=self.pts_to_move)
        if (    self.replace_shape):           centering = True
        if (not self.replace_shape):           centering = False
        env.shape.generate(centering=False)
        env.shape.write_csv()

        try:
            meshed, n_tri = env.shape.mesh(mesh_domain = True,
                                           shape_h     = self.shape_h,
                                           domain_h    = self.domain_h,
                                           xmin        = self.xmin,
                                           xmax        = self.xmax,
                                           ymin        = self.ymin,
                                           ymax        = self.ymax,
                                           mesh_format = 'xml')

            # Do not solve if mesh is too large
            if (n_tri > self.cell_limit):
                meshed = False
                os.system('cp '+env.shape.name+'_'+str(env.shape.index)+'.png '
                          +self.comp_dir+'/save/rejected/.')
        except Exception as exc:
            print(exc)
            meshed = False

        # Generate image
        env.shape.generate_image(plot_pts    = True,
                                 quad_radius = self.max_deformation,
                                 xmin        = self.xmin,
                                 xmax        = self.xmax,
                                 ymin        = self.ymin,
                                 ymax        = self.ymax)

        # Save png and csv files
        os.system('mv '+env.shape.name+'_'+str(env.shape.index)+'.png '
                  +self.comp_dir+'/save/png/.')
        os.system('mv '+env.shape.name+'_'+str(env.shape.index)+'.csv '
                  +self.comp_dir+'/save/csv/.')

        # Copy new shape files to save folder
        if (meshed):
            os.system('cp '+env.shape.name+'_'+str(env.shape.index)+'.xml '
                      +self.comp_dir+'/save/xml/.')

        # Update control number
        env.control_nb += 1

        # Compute reward with try/catch
        self.compute_reward(meshed)

        # Save quantities of interest
        self.save_qoi()

        # Fill next state
        next_state = self.fill_next_state(meshed, env.shape.index)

        # Copy u, v and p solutions to repo
        if (meshed):
            os.system('mv '+str(env.shape.index)+'_u.png '+self.comp_dir+'/save/sol/.')
            os.system('mv '+str(env.shape.index)+'_v.png '+self.comp_dir+'/save/sol/.')
            os.system('mv '+str(env.shape.index)+'_p.png '+self.comp_dir+'/save/sol/.')        
        # Remove mesh file from repo
        if (meshed):
            os.system('rm '+env.shape.name+'_'+str(env.shape.index)+'.xml')

        # Return
        terminal = False
        print("good epoch; reward: {}".format(self.reward[-1]))
        return(next_state, terminal, self.reward[-1])

    def compute_reward(self, meshed):
        # If meshing was successful, reward is computed normally
        if (meshed):
            try:
                # Compute drag and lift
                name = self.comp_dir+'/'+env.shape.name+'_'+str(env.shape.index)+'.xml'
                drag, lift, solved = solve_flow(mesh_file  = name,
                                                final_time = self.final_time,
                                                reynolds   = self.reynolds,
                                                output     = self.output,
                                                cfl        = self.cfl,
                                                pts_x      = env.shape.control_pts[:,0],
                                                pts_y      = env.shape.control_pts[:,1],
                                                xmin       = self.xmin,
                                                xmax       = self.xmax,
                                                ymin       = self.ymin,
                                                ymax       = self.ymax)
                # Save solution png
                os.system('mv '+str(env.shape.index)+'.png '+self.comp_dir+'/save/sol/.')
            except Exception as exc:
                print(exc)
                solved = False

            # If solver was successful
            if (solved):
                # Drag is always <0 while lift changes sign
                penal  = 0.0
                lift   =-lift # Make lift positive
                if (lift > 2.0): lift=2.0*lift # Shaping for faster convergence
                reward = lift/abs(drag)
                reward = max(reward, -10.0)

            # If solver was not successful
            else:
                drag   =-1.0
                lift   = 0.0
                reward =-5.0
                penal  = 5.0

        # If meshing was not successful, we just return a high penalization
        else:
            drag   =-1.0
            lift   = 0.0
            reward =-5.0
            penal  = 5.0

        # Save drag, lift, reward and penalization
        self.drag   = np.append(self.drag,   drag)
        self.lift   = np.append(self.lift,   lift)
        self.reward = np.append(self.reward, reward)
        self.penal  = np.append(self.penal,  penal)

        val_drag   = np.sum(self.drag)/env.shape.index
        val_lift   = np.sum(self.lift)/env.shape.index
        val_reward = np.sum(self.reward)/env.shape.index
        self.avg_drag   = np.append(self.avg_drag,   val_drag)
        self.avg_lift   = np.append(self.avg_lift,   val_lift)
        self.avg_reward = np.append(self.avg_reward, val_reward)

    def save_qoi(self):
        # Retrieve current index
        i = env.shape.index

        # Write drag/lift values to file
        filename = self.comp_dir+'/save/drag_lift'
        with open(filename, 'a') as f:
            f.write('{} {} {} {} {}\n'.format(i,
                                              self.drag[-1],
                                              self.lift[-1],
                                              self.avg_drag[-1],
                                              self.avg_lift[-1]))

        # Write reward and penalization to file
        filename = self.comp_dir+'/save/reward_penalization'
        with open(filename, 'a') as f:
            f.write('{} {} {} {}\n'.format(i,
                                           self.reward[-1],
                                           self.penal[-1],
                                           self.avg_reward[-1]))

    def fill_next_state(self, meshed, index):
        next_state = np.array([])
        for i in range(0,env.shape.n_control_pts):
            next_state = np.append(next_state,env.shape.control_pts[i,0])
            next_state = np.append(next_state,env.shape.control_pts[i,1])
            next_state = np.append(next_state,env.shape.edgy[i])

        return next_state

    @property
    def states(self):
        return dict(
            type='float',
            shape=(3*env.shape.n_control_pts))

    @property
    def actions(self):
        return dict(
            type='float',
            shape=(self.nb_pts_to_move*3),
            min_value=-1.0,
            max_value= 1.0)
