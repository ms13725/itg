import numpy as np
from scipy.special import expit as sigmoid
from simsopt._core.optimizable import Optimizable
from simsopt.mhd import Vmec, vmec_compute_geometry
from simsopt.util import MpiPartition
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve


class MultiSurfaceGyrokineticObjective(Optimizable):
    def __init__(self, vmec, n_surfaces=4, n_theta=32, n_phi=32):
        self.vmec = vmec

        # Define the radial targets
        self.s_targets = np.linspace(0.2, 0.8, n_surfaces)

        # Define the angular resolution for the 2D surface grid
        self.n_theta = n_theta
        self.n_phi = n_phi

        super().__init__(depends_on=[vmec])

    def compute_surfaces(self):
        """Extracts all features and target metrics across full 2D flux surfaces."""
        self.vmec.run()

        # Generate the spatial coordinate arrays
        nfp = self.vmec.wout.nfp
        s_1d = self.s_targets
        theta_1d = np.linspace(0, 2 * np.pi, self.n_theta, endpoint=False)
        phi_1d = np.linspace(0, 2 * np.pi / nfp, self.n_phi, endpoint=False)

        # Compute the geometry at these specific coordinates
        geom = vmec_compute_geometry(
            self.vmec,
            s=s_1d,
            theta=theta_1d,
            phi=phi_1d
        )

        # Global normalization variables
        L_ref = self.vmec.wout.Aminor_p
        psi_a = abs(self.vmec.wout.phi[-1] / (2 * np.pi))
        sign_psi_a = np.sign(self.vmec.wout.phi[-1])

        target_metrics = []

        # Iterate through the surfaces
        for i, s_val in enumerate(s_1d):
            iota_val = np.interp(s_val, self.vmec.s_half_grid, self.vmec.wout.iotas[1:])

            # Extract the 2D surface data for this specific radial depth
            f1 = geom.modB[i, :, :]
            grad_s2 = geom.grad_s_dot_grad_s[i, :, :]

            B_cross_kappa_psi = geom.B_cross_kappa_dot_grad_psi[i, :, :]
            B_cross_kappa_alpha = geom.B_cross_kappa_dot_grad_alpha[i, :, :]

            # Additional arrays required for f2, f3, f6, f7
            B_cross_grad_B_psi = geom.B_cross_grad_B_dot_grad_psi[i, :, :]
            B_cross_grad_B_alpha = geom.B_cross_grad_B_dot_grad_alpha[i, :, :]

            grad_alpha_psi = geom.grad_alpha_dot_grad_psi[i, :, :]
            grad_alpha2 = geom.grad_alpha_dot_grad_alpha[i, :, :]

            # Use PEST poloidal angle for straight-field-line coordinates
            theta_mesh = geom.theta_pest[i, :, :]
            phi_mesh = geom.phi[i, :, :]

            # Manually construct the field line label grid
            alpha = theta_mesh - iota_val * phi_mesh

            # assembling features
            f2 = (L_ref / (2 * np.sqrt(s_val) * psi_a)) * B_cross_grad_B_psi / (f1 ** 3)

            f3 = -L_ref * sign_psi_a * (
                    (alpha / (2 * np.sqrt(s_val) * psi_a)) * B_cross_grad_B_psi +
                    np.sqrt(s_val) * B_cross_grad_B_alpha
            ) / (f1 ** 3)

            f4 = -L_ref * sign_psi_a * (
                    (alpha / (2 * np.sqrt(s_val) * psi_a)) * B_cross_kappa_psi +
                    np.sqrt(s_val) * B_cross_kappa_alpha
            ) / (f1 ** 2)

            f5 = (L_ref ** 2 / (4 * s_val)) * grad_s2

            f6 = -L_ref ** 2 * sign_psi_a * (
                    (alpha / (4 * s_val)) * grad_s2 +
                    0.5 * grad_alpha_psi / psi_a
            )

            f7 = L_ref ** 2 * (
                    (alpha ** 2 / (4 * s_val)) * grad_s2 +
                    (alpha / psi_a) * grad_alpha_psi +
                    s_val * grad_alpha2
            )

            target_1 = np.var(sigmoid((f1 ** 2) * f4) * f5 / f1)
            target_2 = np.mean(sigmoid((f1 ** 2) * f4) * (f5 ** 4) / (f1 ** 2))
            target_3 = np.mean(sigmoid((f1 ** 2) * f4) * (f5 ** 2) / f1)
            target_4 = np.var(sigmoid((f1 ** 2) * f4) * (f5 ** 2) / f1)
            target_5 = np.mean(sigmoid((f1 ** 2) * f4) * (f5 ** 2) / (f1 ** 2))

            target_metrics.append(target_1)

        return np.array(target_metrics)

    def J(self):
        return self.compute_surfaces()


mpi = MpiPartition()

max_mode = 1
v = Vmec('input.LandremanPaul2021_QA_lowres', mpi)
surf = v.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")  # Major radius

gk_multi_obj = MultiSurfaceGyrokineticObjective(v)
prob = LeastSquaresProblem.from_sigma([0.0, 6.0], [1.0, 0.1], [gk_multi_obj.J, v.aspect])

print(f"Initial Multi-surface Metric: {gk_multi_obj.J()}")
least_squares_mpi_solve(prob, mpi, grad=True)
print(f"Final Multi-surface Metric: {gk_multi_obj.J()}")