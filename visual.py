import numpy as np
from simsopt.geo import SurfaceRZFourier

wout_file = 'wout_LandremanPaul2021_QA_lowres_000_000181.nc'

nphi = 64
ntheta = 64
quadpoints_phi = np.linspace(0, 1, nphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)

# 3. Extract the outer boundary surface (s=1.0)
surf = SurfaceRZFourier.from_wout(
    wout_file,
    s=1.0,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

surf.to_vtk("optimized_boundary")
print(f"Successfully generated optimized_boundary.vts from {wout_file}")