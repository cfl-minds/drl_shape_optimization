# Custom imports
from shapes_utils import *
from meshes_utils import *

### ************************************************
### Main execution
### Parameters controlling random shape generation and meshing
### radius         : local radius of curve around control points, in [0,1]   (maximal sharpness  for radius = 1)
### edgy           : controls the smoothness of the curve, in [0,1] (maximal smoothness for edgy   = 0)
### n_pts          : number of random points joined together by Bezier curves
### n_sampling_pts : number of sampled points on each Bezier curve joining two control points
### plot_pts       : True to plot the position of control points in shape image
n_pts          = 4
n_sampling_pts = 10
radius         = 0.5*np.ones([n_pts])
edgy           = 0.5*np.ones([n_pts])
plot_pts       = True
filename       = '3359.csv'
cylinder       = False

# Generate and mesh shape
shape = Shape(filename,None,n_pts,n_sampling_pts,radius,edgy)
#shape = Shape()
shape.read_csv(filename)
shape.generate()
#shape.generate(cylinder=cylinder)
shape.mesh(mesh_domain = True,
           shape_h     = 1.0,
           domain_h    = 1.0,
           xmin        =-10.0,
           xmax        = 20.0,
           ymin        =-10.0,
           ymax        = 10.0,
           mesh_format = 'mesh')
shape.generate_image(plot_pts=plot_pts)
shape.write_csv()
