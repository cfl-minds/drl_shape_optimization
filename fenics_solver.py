# Generic imports
import os
import sys
import math
#import time
import numpy               as np
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec

# Custom imports
from dolfin    import *
from mshr      import *

n_call = 0
u_0    = None
V_0    = None

def solve_flow(*args, **kwargs):
    # Handle optional arguments
    mesh_file   = kwargs.get('mesh_file',   'shape.xml')
    output      = kwargs.get('output',      False)
    final_time  = kwargs.get('final_time',  15.0)
    reynolds    = kwargs.get('reynolds',    10.0)
    pts_x       = kwargs.get('pts_x',       np.array([]))
    pts_y       = kwargs.get('pts_y',       np.array([]))
    cfl         = kwargs.get('cfl',         0.5)
    xmin        = kwargs.get('xmin',       -15.0)
    xmax        = kwargs.get('xmax',       30.0)
    ymin        = kwargs.get('ymin',       -15.0)
    ymax        = kwargs.get('ymax',        15.0)

    # Parameters
    v_in      = 1.0
    mu        = 1.0/reynolds
    rho       = 1.0
    tag_shape = 5
    x_shape   = 4.0
    y_shape   = 4.0
    sol_file  = 'shape.pvd'

    # Create subdomain containing shape boundary
    class Obstacle(SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and
                    (-x_shape < x[0] < x_shape) and
                    (-y_shape < x[1] < y_shape))

    # Define symmetric gradient
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2.0*mu*epsilon(u) - p*Identity(len(u))

    # Compute drag and lift
    def compute_drag_lift(u, p, mu, normal, gamma):
        eps      = 0.5*(nabla_grad(u) + nabla_grad(u).T)
        sigma    = 2.0*mu*eps - p*Identity(len(u))
        traction = dot(sigma, normal)

        forceX   = traction[0]*gamma
        forceY   = traction[1]*gamma
        fX       = assemble(forceX)
        fY       = assemble(forceY)

        return (fX, fY)

    # Create mesh
    # Ugly hack : change dim=3 to dim=2 in xml mesh file
    os.system("sed -i 's/dim=\"3\"/dim=\"2\"/g' "+mesh_file)
    mesh = Mesh(mesh_file)
    h    = mesh.hmin()

    # Compute timestep and max nb of steps
    dt        = cfl*h/v_in
    timestep  = dt
    T         = final_time    
    num_steps = math.floor(T/dt)    

    # Define output solution file
    vtkfile = File(sol_file)
    
    # Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace      (mesh, 'P', 1)

    # Define boundaries
    inflow  = 'near(x[0], '+str(math.floor(xmin))+')'
    outflow = 'near(x[0], '+str(math.floor(xmax))+')'
    wall1   = 'near(x[1], '+str(math.floor(ymin))+')'
    wall2   = 'near(x[1], '+str(math.floor(ymax))+')'
    shape   = 'on_boundary && x[0]>(-'+str(x_shape)+') && x[0]<'+str(x_shape)+' && x[1]>(-'+str(y_shape)+') && x[1]<('+str(y_shape)+')'

    # Define boundary conditions
    bcu_inflow  = DirichletBC(V,        Constant((v_in, 0.0)), inflow)
    bcu_wall1   = DirichletBC(V.sub(1), Constant(0.0),         wall1)
    bcu_wall2   = DirichletBC(V.sub(1), Constant(0.0),         wall2)
    bcu_aile    = DirichletBC(V,        Constant((0.0, 0.0)),  shape)
    bcp_outflow = DirichletBC(Q,        Constant(0.0),         outflow)
    bcu         = [bcu_inflow, bcu_wall1, bcu_wall2, bcu_aile]
    bcp         = [bcp_outflow]

    # Tag shape boundaries for drag_lift computation
    obstacle    = Obstacle()
    boundaries  = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    obstacle.mark(boundaries, tag_shape)
    ds          = Measure('ds', subdomain_data=boundaries)
    gamma_shape = ds(tag_shape)

    # Define trial and test functions
    u, v = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)
    
    # Define functions for solutions at previous and current time steps
    u_n, u_, u_m = Function(V), Function(V), Function(V)
    p_n, p_      = Function(Q), Function(Q)

    # Define initial value
    global n_call
    global u_0

    if (n_call != 0):
        u_0.set_allow_extrapolation(True)
        u_n  = project(u_0, V)
        show = True
    else:
        show = False

    # Define expressions and constants used in variational forms
    U   = 0.5*(u_n + u)
    n   = FacetNormal(mesh)
    f   = Constant((0, 0))
    dt  = Constant(dt)
    mu  = Constant(mu)
    rho = Constant(rho)

    # Set BDF2 coefficients for 1st timestep
    bdf2_a = Constant( 1.0)
    bdf2_b = Constant(-1.0)
    bdf2_c = Constant( 0.0)

    # Define variational problem for step 1
    # Using BDF2 scheme
    F1 = dot((bdf2_a*u + bdf2_b*u_n + bdf2_c*u_m)/dt, v)*dx + dot(dot(u_n, nabla_grad(u)), v)*dx + inner(sigma(u, p_n), epsilon(v))*dx + dot(p_n*n, v)*ds - dot(mu*nabla_grad(u)*n, v)*ds - dot(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Define variational problem for step 2
    a2 = dot(nabla_grad(p),   nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (bdf2_a/dt)*div(u_)*q*dx

    # Define variational problem for step 3
    a3 = dot(u,  v)*dx
    L3 = dot(u_, v)*dx - (dt/bdf2_a)*dot(nabla_grad(p_ - p_n), v)*dx

    # Assemble A3 matrix since it will not need re-assembly
    A3 = assemble(a3)

    # Initialize drag and lift
    drag      = 0.0
    lift      = 0.0
    lfdr      = 0.0
    drag_inst = np.array([])
    lift_inst = np.array([])
    lfdr_inst = np.array([])
    drag_avg  = np.array([])
    lift_avg  = np.array([])
    lfdr_avg  = np.array([])

    ########################################
    # Time-stepping loop
    ########################################
    try:
        k     = 0
        t     = 0.0
        t_arr = np.array([])
        for m in range(num_steps):
            # Update current time
            t += timestep

            # Step 1: Tentative velocity step
            A1 = assemble(a1)
            b1 = assemble(L1)
            [bc.apply(A1) for bc in bcu]
            [bc.apply(b1) for bc in bcu]
            solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg') #gmres

            # Step 2: Pressure correction step
            A2 = assemble(a2)
            b2 = assemble(L2)
            [bc.apply(A2) for bc in bcp]
            [bc.apply(b2) for bc in bcp]
            solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

            # Step 3: Velocity correction step
            b3 = assemble(L3)
            solve(A3, u_.vector(), b3, 'cg',       'sor')

            # Update previous solution
            u_m.assign(u_n)
            u_n.assign(u_)
            p_n.assign(p_)

            # Compute and store drag and lift
            avg_start_it = math.floor(num_steps/2)
            if (m > avg_start_it):
                (fX,fY) = compute_drag_lift(u_, p_, mu, n, gamma_shape)
                drag_inst = np.append(drag_inst, fX)
                lift_inst = np.append(lift_inst, fY)
                lfdr_inst = np.append(lfdr_inst, fY/fX)
                drag     += fX
                lift     += fY
                lfdr     += fY/fX
                drag_avg  = np.append(drag_avg, drag/(k+1))
                lift_avg  = np.append(lift_avg, lift/(k+1))
                lfdr_avg  = np.append(lfdr_avg, lfdr/(k+1))
                t_arr     = np.append(t_arr, t)

                # Increment local counter
                k += 1

            # Set BDF2 coefficients for m>1
            bdf2_a.assign(Constant( 3.0/2.0))
            bdf2_b.assign(Constant(-2.0))
            bdf2_c.assign(Constant( 1.0/2.0))

#                # Check execution time
#                end     = time.time()
#                elapsed = end - start

#            if (elapsed > max_time):
#                return 0.0, 0.0, False
            ########################################

        # Ouput field maps if necessary
        if (output):
            start_it = 0
            uu       = interpolate(u_n, V)
            pp       = interpolate(p_n, Q)
            u        = uu.sub(0)
            v        = uu.sub(1)

            filename = mesh_file.split('_')[-1]
            filename = filename.split('.')[0]        
        
            plt.figure()
            plt.subplot2grid((5,4),(0,0),colspan=2,rowspan=2).set_title('p')
            plot(pp)
            plt.subplot2grid((5,4),(0,2),colspan=2,rowspan=2).set_title('u')
            plot(uu)
            plt.subplot2grid((5,4),(2,0),colspan=2,rowspan=1).set_title('inst_drag')
            plt.plot(t_arr[start_it:],drag_inst[start_it:])
            plt.subplot2grid((5,4),(2,2),colspan=2,rowspan=1).set_title('inst_lift')
            plt.plot(t_arr[start_it:],lift_inst[start_it:])
            plt.subplot2grid((5,4),(3,0),colspan=2,rowspan=1).set_title('avg_drag')
            plt.plot(t_arr[start_it:],drag_avg[start_it:])
            plt.subplot2grid((5,4),(3,2),colspan=2,rowspan=1).set_title('avg_lift')
            plt.plot(t_arr[start_it:],lift_avg[start_it:])
            plt.subplot2grid((5,4),(4,0),colspan=2,rowspan=1).set_title('inst_lift/drag')
            plt.plot(t_arr[start_it:],lfdr_inst[start_it:])
            plt.subplot2grid((5,4),(4,2),colspan=2,rowspan=1).set_title('avg_lift/drag')
            plt.plot(t_arr[start_it:],lfdr_avg[start_it:])
            plt.tight_layout()
            plt.savefig(filename+'.png', dpi=400)
            plt.close()

            plot_pts = False
            if (len(pts_x) > 0): plot_pts = True

            fig = plt.figure()
            plot(u, range_min=-v_in, range_max=v_in)
            plt.axis('off')
            ax = plt.gca()
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if (plot_pts): plt.scatter(pts_x,pts_y,color='k',s=32)
            plt.savefig(filename+'_u.png', dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()

            fig = plt.figure()
            plot(v, range_min=-v_in, range_max=v_in)
            plt.axis('off')
            ax = plt.gca()
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)        
            if (plot_pts): plt.scatter(pts_x,pts_y,color='k',s=32)
            plt.savefig(filename+'_v.png', dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()

            fig = plt.figure()
            plot(pp, range_min=-v_in, range_max=v_in)
            plt.axis('off')
            ax = plt.gca()
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)                
            if (plot_pts): plt.scatter(pts_x,pts_y,color='k',s=32)
            plt.savefig(filename+'_p.png', dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()

            # Average drag and lift values
            drag = drag_avg[-1]
            lift = lift_avg[-1]

            print(drag, lift)

            # Save final solutions for next solving
            u_0     = u_n
            n_call += 1
                
    except Exception as exc:
        print(exc)
        return 0.0, 0.0, False            

    return drag, lift, True
