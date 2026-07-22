import numpy as np
from simsopt.mhd import Vmec, vmec_compute_geometry
from simsopt.geo import SurfaceRZFourier

wout_file = 'wout_6_14.nc'
v_final = Vmec(wout_file)

n_surfaces = 10
s_1d = np.linspace(0.1, 1.0, n_surfaces)
surf = SurfaceRZFourier.from_wout(wout_file, s=1.0, ntheta=64, nphi=64)

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

# Extract Outer Surface
f1_outer = f1[-1, :, :]
f2_outer = f2[-1, :, :]
f3_outer = f3[-1, :, :]
f4_outer = f4[-1, :, :]
f5_outer = f5[-1, :, :]
f6_outer = f6[-1, :, :]
f7_outer = f7[-1, :, :]

# Calculate Radial Averages across all surfaces
f1_avg = np.mean(f1, axis=0)
f2_avg = np.mean(f2, axis=0)
f3_avg = np.mean(f3, axis=0)
f4_avg = np.mean(f4, axis=0)
f5_avg = np.mean(f5, axis=0)
f6_avg = np.mean(f6, axis=0)
f7_avg = np.mean(f7, axis=0)

surf.to_vtk(
    "outermost_surface",
    extra_data={
        'f1_outer': f1_outer.T[:, :, None],
        'f2_outer': f2_outer.T[:, :, None],
        'f3_outer': f3_outer.T[:, :, None],
        'f4_outer': f4_outer.T[:, :, None],
        'f5_outer': f5_outer.T[:, :, None],
        'f6_outer': f6_outer.T[:, :, None],
        'f7_outer': f7_outer.T[:, :, None],

        'f1_avg': f1_avg.T[:, :, None],
        'f2_avg': f2_avg.T[:, :, None],
        'f3_avg': f3_avg.T[:, :, None],
        'f4_avg': f4_avg.T[:, :, None],
        'f5_avg': f5_avg.T[:, :, None],
        'f6_avg': f6_avg.T[:, :, None],
        'f7_avg': f7_avg.T[:, :, None]
    }
)
