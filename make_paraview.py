import numpy as np
from scipy.special import expit as sigmoid
from simsopt.mhd import Vmec, vmec_compute_geometry
from simsopt.geo import SurfaceRZFourier


def compute_target_fields(wout_file='wout_final.nc', ntheta=64, nphi=64, n_surfaces=10):
    """
    Compute turbulence targets t1-t5 over the full torus on the boundary's (VMEC theta, phi) grid.

    Returns (surf, fields): surf is the s = 1 SurfaceRZFourier, fields is a dict of (ntheta, nphi) arrays
    't<N>_outer' (value on the boundary) / 't<N>_avg' (mean over s = 0.1..1) for N = 1..5.
    """
    v_final = Vmec(wout_file)

    quadpoints_phi = np.linspace(0, 1, nphi, endpoint=True)
    quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
    surf = SurfaceRZFourier.from_wout(
        wout_file,
        range='full_torus',
        s=1.0,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta
    )

    s_1d = np.linspace(0.1, 1.0, n_surfaces)
    theta_1d = surf.quadpoints_theta * 2 * np.pi
    phi_1d = surf.quadpoints_phi * 2 * np.pi

    geom = vmec_compute_geometry(v_final, s=s_1d, theta=theta_1d, phi=phi_1d)

    L_ref = v_final.wout.Aminor_p
    s_3d = s_1d[:, None, None]

    # Assemble features
    f1 = geom.modB

    f4 = -L_ref * np.sign(v_final.wout.phi[-1]) * (
            np.sqrt(s_3d) * geom.B_cross_kappa_dot_grad_alpha
    ) / (f1 ** 2)

    f5 = (L_ref ** 2 / (4 * s_3d)) * geom.grad_s_dot_grad_s

    weight = sigmoid((f1 ** 2) * f4)

    # variance targets
    t1 = weight * f5 / f1
    t4 = weight * (f5 ** 2) / f1

    # squared deviations use mean of each surface, not mean of whole 3d array
    t1_deviation = (t1 - np.mean(t1, axis=(1, 2), keepdims=True)) ** 2
    t4_deviation = (t4 - np.mean(t4, axis=(1, 2), keepdims=True)) ** 2

    # mean targets
    t2 = weight * (f5 ** 4) / (f1 ** 2)
    t3 = weight * (f5 ** 2) / f1
    t5 = weight * (f5 ** 2) / (f1 ** 2)

    fields = {}
    for n, t in enumerate([t1_deviation, t2, t3, t4_deviation, t5], start=1):
        fields[f't{n}_outer'] = t[-1, :, :]                  # outer surface
        fields[f't{n}_avg'] = np.mean(t, axis=0)             # radial average across surfaces
    return surf, fields


if __name__ == '__main__':
    filename = 'wout_final.nc'
    surf, fields = compute_target_fields(filename)
    extra_data = {name: t.T[:, :, None] for name, t in fields.items()}
    surf.to_vtk("objectives_paraview", extra_data=extra_data)
