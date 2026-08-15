import glob
import re
import sys
import argparse
import numpy as np
from scipy.special import expit as sigmoid
from simsopt.mhd import Vmec, vmec_compute_geometry
from simsopt.geo import SurfaceRZFourier

parser = argparse.ArgumentParser(description="Generate Paraview file from the latest VMEC wout file.")
parser.add_argument("prefix", type=str, help="Prefix of the wout file")
args = parser.parse_args()

file_pattern = f"wout_{args.prefix}_*.nc"
files = glob.glob(file_pattern)

if not files:
    print(f"Error: No files found matching pattern {file_pattern}", file=sys.stderr)
    sys.exit(1)


def extract_suffix(filepath):
    match = re.search(r'_(\d+)\.nc$', filepath)
    return int(match.group(1)) if match else -1


config = {'surface_method': 'from_wout'}
surf_filename = max(files, key=extract_suffix)
print(f'building paraview plot from {surf_filename}')

surf_filename = 'wout_nfp4_QH_warm_start_000_000341.nc'
v_final = Vmec(surf_filename)

# surface_constructor = getattr(SurfaceRZFourier, config["surface_method"])
#
# qphi = 64
# qtheta = 64
# quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
# quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
# surf = surface_constructor(
#     surf_filename,
#     range="full torus",
#     quadpoints_phi=quadpoints_phi,
#     quadpoints_theta=quadpoints_theta
# )

# surf = SurfaceRZFourier.from_wout(surf_filename, s=1.0, ntheta=64, nphi=64)

nphi = 64
ntheta = 64
quadpoints_phi = np.linspace(0, 1, nphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
surf = SurfaceRZFourier.from_wout(
    surf_filename,
    range='full_torus',
    s=1.0,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

n_surfaces = 10
s_1d = np.linspace(0.1, 1.0, n_surfaces)
theta_1d = surf.quadpoints_theta * 2 * np.pi
phi_1d = surf.quadpoints_phi * 2 * np.pi

geom = vmec_compute_geometry(v_final, s=s_1d, theta=theta_1d, phi=phi_1d)

L_ref = v_final.wout.Aminor_p
psi_a = abs(v_final.wout.phi[-1] / (2 * np.pi))
sign_psi_a = np.sign(v_final.wout.phi[-1])

s_3d = s_1d[:, None, None]
iota_1d = np.interp(s_1d, v_final.s_half_grid, v_final.wout.iotas[1:])
iota_3d = iota_1d[:, None, None]

# Extract 3D geometry arrays
grad_s2 = geom.grad_s_dot_grad_s
B_cross_kappa_psi = geom.B_cross_kappa_dot_grad_psi
B_cross_kappa_alpha = geom.B_cross_kappa_dot_grad_alpha
B_cross_grad_B_psi = geom.B_cross_grad_B_dot_grad_psi
B_cross_grad_B_alpha = geom.B_cross_grad_B_dot_grad_alpha
grad_alpha_psi = geom.grad_alpha_dot_grad_psi
grad_alpha2 = geom.grad_alpha_dot_grad_alpha

# Construct the 3D field line label grid
theta_mesh = geom.theta_pest
phi_mesh = geom.phi
alpha = theta_mesh - iota_3d * phi_mesh

# Assemble features
f1 = geom.modB

f2 = (L_ref / (2 * np.sqrt(s_3d) * psi_a)) * B_cross_grad_B_psi / (f1 ** 3)

f3 = -L_ref * sign_psi_a * (
        (alpha / (2 * np.sqrt(s_3d) * psi_a)) * B_cross_grad_B_psi +
        np.sqrt(s_3d) * B_cross_grad_B_alpha
) / (f1 ** 3)

f4 = -L_ref * sign_psi_a * (
        (alpha / (2 * np.sqrt(s_3d) * psi_a)) * B_cross_kappa_psi +
        np.sqrt(s_3d) * B_cross_kappa_alpha
) / (f1 ** 2)

f5 = (L_ref ** 2 / (4 * s_3d)) * grad_s2

f6 = -L_ref ** 2 * sign_psi_a * (
        (alpha / (4 * s_3d)) * grad_s2 +
        0.5 * grad_alpha_psi / psi_a
)

f7 = L_ref ** 2 * (
        (alpha ** 2 / (4 * s_3d)) * grad_s2 +
        (alpha / psi_a) * grad_alpha_psi +
        s_3d * grad_alpha2
)

# variance targets
t1 = sigmoid((f1 ** 2) * f4) * f5 / f1
t4 = sigmoid((f1 ** 2) * f4) * (f5 ** 2) / f1

# squared deviations use mean of each surface, not mean of whole 3d array
t1_deviation = (t1 - np.mean(t1, axis=(1, 2), keepdims=True)) ** 2
t4_deviation = (t4 - np.mean(t4, axis=(1, 2), keepdims=True)) ** 2

# mean targets
t2 = sigmoid((f1 ** 2) * f4) * (f5 ** 4) / (f1 ** 2)
t3 = sigmoid((f1 ** 2) * f4) * (f5 ** 2) / f1
t5 = sigmoid((f1 ** 2) * f4) * (f5 ** 2) / (f1 ** 2)

# outer surfaces
t1_outer = t1_deviation[-1, :, :]
t2_outer = t2[-1, :, :]
t3_outer = t3[-1, :, :]
t4_outer = t4_deviation[-1, :, :]
t5_outer = t5[-1, :, :]

# radial averages across surfaces
t1_avg = np.mean(t1_deviation, axis=0)
t2_avg = np.mean(t2, axis=0)
t3_avg = np.mean(t3, axis=0)
t4_avg = np.mean(t4_deviation, axis=0)
t5_avg = np.mean(t5, axis=0)

surf.to_vtk(
    "objectives_nfp4_test",
    extra_data={
        't1_outer': t1_outer.T[:, :, None],
        't2_outer': t2_outer.T[:, :, None],
        't3_outer': t3_outer.T[:, :, None],
        't4_outer': t4_outer.T[:, :, None],
        't5_outer': t5_outer.T[:, :, None],

        't1_avg': t1_avg.T[:, :, None],
        't2_avg': t2_avg.T[:, :, None],
        't3_avg': t3_avg.T[:, :, None],
        't4_avg': t4_avg.T[:, :, None],
        't5_avg': t5_avg.T[:, :, None]
    }
)
