# Generic imports
import os, os.path
import math
import pygmsh
import meshio
import scipy.special
import matplotlib
import numpy             as np
import matplotlib.pyplot as plt

# Custom imports
from meshes_utils import *

### ************************************************
### Class defining shape object
class Shape:
    ### ************************************************
    ### Constructor
    def __init__(self,
                 name          ='shape',
                 control_pts   =None,
                 n_control_pts =None,
                 n_sampling_pts=None,
                 radius        =None,
                 edgy          =None):
        if (name           is None): name           = 'shape'
        if (control_pts    is None): control_pts    = np.array([])
        if (n_control_pts  is None): n_control_pts  = 0
        if (n_sampling_pts is None): n_sampling_pts = 0
        if (radius         is None): radius         = np.array([])
        if (edgy           is None): edgy           = np.array([])

        self.name           = name
        self.control_pts    = control_pts
        self.n_control_pts  = n_control_pts
        self.n_sampling_pts = n_sampling_pts
        self.curve_pts      = np.array([])
        self.area           = 0.0
        self.index          = 0

        if (len(radius) == n_control_pts): self.radius = radius
        if (len(radius) == 1):             self.radius = radius*np.ones([n_control_pts])

        if (len(edgy) == n_control_pts):   self.edgy = edgy
        if (len(edgy) == 1):               self.edgy = edgy*np.ones([n_control_pts])

        subname             = name.split('_')
        if (len(subname) == 2): # name is of the form shape_?.xxx
            self.name       = subname[0]
            index           = subname[1].split('.')[0]
            self.index      = int(index)
        if (len(subname) >  2): # name contains several '_'
            print('Please do not use several "_" char in shape name')
            quit()

        if (len(control_pts) > 0):
            self.control_pts   = control_pts
            self.n_control_pts = len(control_pts)

    ### ************************************************
    ### Reset object
    def reset(self):
        self.name           = 'shape'
        self.control_pts    = np.array([])
        self.n_control_pts  = 0
        self.n_sampling_pts = 0
        self.radius         = np.array([])
        self.edgy           = np.array([])
        self.curve_pts      = np.array([])
        self.area           = 0.0

    ### ************************************************
    ### Generate shape
    def generate(self, *args, **kwargs):
        # Handle optional argument
        centering = kwargs.get('centering', True)
        cylinder  = kwargs.get('cylinder',  False)
        magnify   = kwargs.get('magnify',   1.0)

        # Generate random control points if empty
        if (len(self.control_pts) == 0):
            if (cylinder):
                self.control_pts = generate_cylinder_pts(self.n_control_pts)
            else:
                self.control_pts = generate_random_pts(self.n_control_pts)

        # Magnify
        self.control_pts *= magnify

        # Center set of points
        if (centering):
            center            = np.mean(self.control_pts, axis=0)
            self.control_pts -= center

        # Sort points counter-clockwise
        #control_pts, radius, edgy = ccw_sort(self.control_pts,
        #                                     self.radius,
        #                                     self.edgy)
        control_pts = np.array(self.control_pts)
        radius = np.array(self.radius)
        edgy = np.array(self.edgy)

        #self.control_pts = control_pts
        #self.radius      = radius
        #self.edgy        = edgy

        # Create copy of control_pts for further modification
        augmented_control_pts = control_pts

        # Add first point as last point to close curve
        augmented_control_pts = np.append(augmented_control_pts,
                                          np.atleast_2d(augmented_control_pts[0,:]), axis=0)

        # Compute list of cartesian angles from one point to the next
        vector = np.diff(augmented_control_pts, axis=0)
        angles = np.arctan2(vector[:,1],vector[:,0])
        wrap   = lambda angle: (angle >= 0.0)*angle + (angle < 0.0)*(angle+2*np.pi)
        angles = wrap(angles)

        # Create a second list of angles shifted by one point
        # to compute an average of the two at each control point.
        # This helps smoothing the curve around control points
        angles1 = angles
        angles2 = np.roll(angles,1)

        angles  = edgy*angles1 + (1.0-edgy)*angles2 + (np.abs(angles2-angles1) > np.pi)*np.pi

        # Add first angle as last angle to close curve
        angles  = np.append(angles, [angles[0]])

        # Compute curve segments
        local_curves = []
        for i in range(0,len(augmented_control_pts)-1):
            local_curve = generate_bezier_curve(augmented_control_pts[i,:],
                                                augmented_control_pts[i+1,:],
                                                angles[i],
                                                angles[i+1],
                                                self.n_sampling_pts,
                                                radius[i])
            local_curves.append(local_curve)

        curve          = np.concatenate([c for c in local_curves])
        x, y           = curve.T
        z              = np.zeros(x.size)
        self.curve_pts = np.column_stack((x,y,z))
        self.curve_pts = remove_duplicate_pts(self.curve_pts)

        # Compute area
        self.compute_area()

    ### ************************************************
    ### Write image
    def generate_image(self, *args, **kwargs):
        # Handle optional argument
        special_pt     = kwargs.get('special_pt',     None)
        plot_pts       = kwargs.get('plot_pts',       False)
        override_name  = kwargs.get('override_name',  '')
        show_quadrants = kwargs.get('show_quadrants', 'False')
        quad_radius    = kwargs.get('quad_radius',    1.0)
        xmin           = kwargs.get('xmin',          -5.0)
        xmax           = kwargs.get('xmax',          10.0)
        ymin           = kwargs.get('ymin',          -5.0)
        ymax           = kwargs.get('ymax',           5.0)

        # Plot shape
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.fill(self.curve_pts[:,0],
                 self.curve_pts[:,1],
                 'black',
                 linewidth=2.5)

        # Plot points
        # Each point gets a different color
        if (plot_pts):
            colors = matplotlib.cm.ocean(np.linspace(0,1,self.n_control_pts))
            plt.scatter(self.control_pts[:,0],
                        self.control_pts[:,1],
                        color=colors)

        # Plot special point
        if (special_pt) is not None:
            plt.plot(self.control_pts[special_pt,0],
                     self.control_pts[special_pt,1],
                     'o',
                     color=(1.0,0.494,0.180))

        # Plot quadrants
        if (show_quadrants):
            for pt in range(self.n_control_pts):
                dangle = (360.0/float(self.n_control_pts))
                angle  = dangle*float(pt)+dangle/2.0
                x      = quad_radius*math.cos(math.radians(angle))
                y      = quad_radius*math.sin(math.radians(angle))
                plt.plot([0, x],[0, y],color='w')

            circle = plt.Circle((0,0),quad_radius,fill=False,color='w')
            plt.gcf().gca().add_artist(circle)

        # Save image
        filename = self.name+'_'+str(self.index)+'.png'
        if (override_name != ''): filename = override_name

        plt.savefig(filename,
                    dpi=200,
                    bbox_inches='tight',
                    pad_inches=0,
                    facecolor=(0.784,0.773,0.741))
        plt.clf()

    ### ************************************************
    ### Write csv
    def write_csv(self):
        with open(self.name+'_'+str(self.index)+'.csv','w') as file:
            # Write header
            file.write('{} {}\n'.format(self.n_control_pts,
                                              self.n_sampling_pts))

            # Write radii
            for i in range(0,self.n_control_pts):
                file.write('{}\n'.format(self.radius[i]))

            # Write edgy
            for i in range(0,self.n_control_pts):
                file.write('{}\n'.format(self.edgy[i]))

            # Write control points coordinates
            for i in range(0,self.n_control_pts):
                file.write('{} {}\n'.format(self.control_pts[i,0],
                                            self.control_pts[i,1]))

    ### ************************************************
    ### Read csv and initialize shape with it
    def read_csv(self, filename, *args, **kwargs):
        # Handle optional argument
        keep_numbering = kwargs.get('keep_numbering', False)

        if (not os.path.isfile(filename)):
            print('I could not find csv file: '+filename)
            print('Exiting now')
            exit()

        self.reset()
        sfile  = filename.split('.')
        sfile  = sfile[-2]
        sfile  = sfile.split('/')
        name   = sfile[-1]

        if (keep_numbering):
            sname = name.split('_')
            name  = sname[0]
            name  = name+'_'+str(self.index)

        x      = []
        y      = []
        radius = []
        edgy   = []

        with open(filename) as file:
            header         = file.readline().split()
            n_control_pts  = int(header[0])
            n_sampling_pts = int(header[1])

            for i in range(0,n_control_pts):
                rad = file.readline().split()
                radius.append(float(rad[0]))

            for i in range(0,n_control_pts):
                edg = file.readline().split()
                edgy.append(float(edg[0]))

            for i in range(0,n_control_pts):
                coords = file.readline().split()
                x.append(float(coords[0]))
                y.append(float(coords[1]))
                control_pts = np.column_stack((x,y))

        self.__init__(name,
                      control_pts,
                      n_control_pts,
                      n_sampling_pts,
                      radius,
                      edgy)

    ### ************************************************
    ### Mesh shape
    def mesh(self, *args, **kwargs):
        # Handle optional argument
        mesh_domain = kwargs.get('mesh_domain', False)
        xmin        = kwargs.get('xmin',       -5.0)
        xmax        = kwargs.get('xmax',        10.0)
        ymin        = kwargs.get('ymin',       -5.0)
        ymax        = kwargs.get('ymax',        5.0)
        shape_h     = kwargs.get('shape_h',     10.0)
        domain_h    = kwargs.get('domain_h',    20.0)
        mesh_format = kwargs.get('mesh_format', 'mesh')

        # Convert curve to polygon
        mesh_size = 1.0
        min_size  = mesh_size*shape_h
        geom      = pygmsh.built_in.Geometry()
        poly      = geom.add_polygon(self.curve_pts,
                                     mesh_size*shape_h,
                                     make_surface=not mesh_domain)

        # Mesh domain if necessary
        if (mesh_domain):
            # Compute an intermediate mesh size
            border = geom.add_rectangle(xmin, xmax,
                                        ymin, ymax,
                                        0.0,
                                        mesh_size*domain_h,
                                        holes=[poly.line_loop])

        # Generate mesh and write in medit format
        try:
            mesh = pygmsh.generate_mesh(geom,
                                        extra_gmsh_arguments=["-v", "0"])
        except AssertionError:
            print('Meshing failed')
            return False, 0, 0.0
        else:
            # Compute data from mesh
            n_tri = len(mesh.cells['triangle'])

            # Remove vertex keywork from cells dictionnary
            # to avoid warning message from meshio
            del mesh.cells['vertex']

            # Remove lines if output format is xml
            if (mesh_format == 'xml'): del mesh.cells['line']

            # Write mesh
            filename = self.name+'_'+str(self.index)+'.'+mesh_format
            meshio.write_points_cells(filename, mesh.points, mesh.cells)

            return True, n_tri

    ### ************************************************
    ### Get shape bounding box
    def compute_bounding_box(self):
        x_max, y_max = np.amax(self.control_pts,axis=0)
        x_min, y_min = np.amin(self.control_pts,axis=0)

        dx = x_max - x_min
        dy = y_max - y_min

        return dx, dy

    ### ************************************************
    ### Modify shape given a deformation field
    def modify_shape_from_field(self, deformation, *args, **kwargs):
        # Handle optional argument
        replace  = kwargs.get('replace',  False)
        pts_list = kwargs.get('pts_list', [])

        # Check inputs
        if (pts_list == []):
            if (len(deformation[:,0]) != self.n_control_pts):
                print('Input deformation field does not have right length')
                quit()
        if (len(deformation[0,:]) not in [2, 3]):
            print('Input deformation field does not have right width')
            quit()
        if (pts_list != []):
            if (len(pts_list) != len(deformation)):
                print('Lengths of pts_list and deformation are different')
                quit()

        # If shape is to be replaced entirely
        if (    replace):
            # If a list of points is provided
            if (pts_list != []):
                for i in range(len(pts_list)):
                    self.control_pts[pts_list[i],0] = deformation[i,0]
                    self.control_pts[pts_list[i],1] = deformation[i,1]
                    self.edgy[pts_list[i]]          = deformation[i,2]
            # Otherwise
            if (pts_list == []):
                self.control_pts[:,0] = deformation[:,0]
                self.control_pts[:,1] = deformation[:,1]
                self.edgy[:]          = deformation[:,2]
        # Otherwise
        if (not replace):
            # If a list of points to deform is provided
            if (pts_list != []):
                for i in range(len(pts_list)):
                    self.control_pts[pts_list[i],0] += deformation[i,0]
                    self.control_pts[pts_list[i],1] += deformation[i,1]
                    self.edgy[pts_list[i]]          += deformation[i,2]
            # Otherwise
            if (pts_list == []):
                self.control_pts[:,0] += deformation[:,0]
                self.control_pts[:,1] += deformation[:,1]
                self.edgy[:]          += deformation[:,2]

        # Increment shape index
        self.index += 1

    ### ************************************************
    ### Compute shape area
    def compute_area(self):
        self.area = 0.0

        # Use Green theorem to compute area
        for i in range(0,len(self.curve_pts)-1):
            x1 = self.curve_pts[i-1,0]
            x2 = self.curve_pts[i,  0]
            y1 = self.curve_pts[i-1,1]
            y2 = self.curve_pts[i,  1]

            self.area += 2.0*(y1+y2)*(x2-x1)

### End of class Shape
### ************************************************

### ************************************************
### Compute distance between two points
def compute_distance(p1, p2):

    return np.sqrt(np.sum((p2-p1)**2))

### ************************************************
### Generate n_pts random points in the unit square
def generate_random_pts(n_pts):

    return np.random.rand(n_pts,2)

### ************************************************
### Generate cylinder points
def generate_cylinder_pts(n_pts):
    if (n_pts < 4):
        print('Not enough points to generate cylinder')
        exit()

    pts = np.zeros([n_pts, 2])
    ang = 2.0*math.pi/n_pts
    for i in range(0,n_pts):
        pts[i,:] = [math.cos(float(i)*ang),math.sin(float(i)*ang)]

    return pts

### ************************************************
### Compute minimal distance between successive pts in array
def compute_min_distance(pts):
    dist_min = 1.0e20
    for i in range(len(pts)-1):
        p1       = pts[i  ,:]
        p2       = pts[i+1,:]
        dist     = compute_distance(p1,p2)
        dist_min = min(dist_min,dist)

    return dist_min

### ************************************************
### Remove duplicate points in input coordinates array
### WARNING : this routine is highly sub-optimal
def remove_duplicate_pts(pts):
    to_remove = []

    for i in range(len(pts)):
        for j in range(len(pts)):
            # Check that i and j are not identical
            if (i == j):
                continue

            # Check that i and j are not removed points
            if (i in to_remove) or (j in to_remove):
                continue

            # Compute distance between points
            pi = pts[i,:]
            pj = pts[j,:]
            dist = compute_distance(pi,pj)

            # Tag the point to be removed
            if (dist < 1.0e-8):
                to_remove.append(j)

    # Sort elements to remove in reverse order
    to_remove.sort(reverse=True)

    # Remove elements from pts
    for pt in to_remove:
        pts = np.delete(pts, pt, 0)

    return pts

### ************************************************
### Counter Clock-Wise sort
###  - Take a cloud of points and compute its geometric center
###  - Translate points to have their geometric center at origin
###  - Compute the angle from origin for each point
###  - Sort angles by ascending order
def ccw_sort(pts, rad, edg):
    geometric_center = np.mean(pts,axis=0)
    translated_pts   = pts - geometric_center
    angles           = np.arctan2(translated_pts[:,0], translated_pts[:,1])
    x                = angles.argsort()
    pts2             = np.array(pts)
    rad2             = np.array(rad)
    edg2             = np.array(edg)

    return pts2[x,:], rad2[x], edg2[x]

### ************************************************
### Compute Bernstein polynomial value
def compute_bernstein(n,k,t):
    k_choose_n = scipy.special.binom(n,k)

    return k_choose_n * (t**k) * ((1.0-t)**(n-k))

### ************************************************
### Sample Bezier curves given set of control points
### and the number of sampling points
### Bezier curves are parameterized with t in [0,1]
### and are defined with n control points P_i :
### B(t) = sum_{i=0,n} B_i^n(t) * P_i
def sample_bezier_curve(control_pts, n_sampling_pts):
    n_control_pts = len(control_pts)
    t             = np.linspace(0, 1, n_sampling_pts)
    curve         = np.zeros((n_sampling_pts, 2))

    for i in range(n_control_pts):
        curve += np.outer(compute_bernstein(n_control_pts-1, i, t),
                          control_pts[i])

    return curve

### ************************************************
### Generate Bezier curve between two pts
def generate_bezier_curve(p1, p2, angle1, angle2, n_sampling_pts, radius):
    # Sample the curve if necessary
    if (n_sampling_pts != 0):
        dist = compute_distance(p1, p2)
        if (radius == 'random'):
            radius = 0.707*dist*np.random.uniform(low=0.0, high=1.0)
        else:
            radius = 0.707*dist*radius

            # Create array of control pts for cubic Bezier curve
            # First and last points are given, while the two intermediate
            # points are computed from edge points, angles and radius
            control_pts      = np.zeros((4,2))
            control_pts[0,:] = p1[:]
            control_pts[3,:] = p2[:]
            control_pts[1,:] = p1 + np.array(
                [radius*np.cos(angle1), radius*np.sin(angle1)])
            control_pts[2,:] = p2 + np.array(
                [radius*np.cos(angle2+np.pi), radius*np.sin(angle2+np.pi)])

            # Compute points on the Bezier curve
            curve = sample_bezier_curve(control_pts, n_sampling_pts)

    # Else return just a straight line
    else:
        curve = p1
        curve = np.vstack([curve,p2])

    return curve
