import sys
import numpy as np
from simsopt.geo import SurfaceRZFourier

wout_file = sys.argv[1]

nphi = 64
ntheta = 64
quadpoints_phi = np.linspace(0, 1, nphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)

surf = SurfaceRZFourier.from_wout(
    wout_file,
    s=1.0,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

surf.to_vtk(sys.argv[2])
print(f"Successfully generated {sys.argv[2]}.vts from {wout_file}")