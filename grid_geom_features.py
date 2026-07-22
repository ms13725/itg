import numpy as np
from scipy.special import expit as sigmoid
from simsopt._core.optimizable import Optimizable
from simsopt.mhd import vmec_compute_geometry


class MultiSurfaceGyrokineticObjective(Optimizable):
    def __init__(self, vmec, n_surfaces=4, n_theta=16, n_phi=16):
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
            grad_s2 = geom.grad_s_dot_grad_s[i, :, :]
            B_cross_kappa_psi = geom.B_cross_kappa_dot_grad_psi[i, :, :]
            B_cross_kappa_alpha = geom.B_cross_kappa_dot_grad_alpha[i, :, :]
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
            f1 = geom.modB[i, :, :]

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
