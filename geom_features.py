import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.special import expit as sigmoid
from simsopt.util import MpiPartition
from simsopt.mhd import Vmec, vmec_fieldlines
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt._core.optimizable import Optimizable


class MultiTubeGyrokineticObjective(Optimizable):
    def __init__(self, vmec, n_tubes=4):
        self.vmec = vmec
        self.n_tubes = n_tubes

        # SAMPLE ONCE: Fix the radial locations for the duration of the optimization
        # to ensure finite-difference gradients are smooth and stable.
        np.random.seed(42)
        self.rhos = np.random.uniform(0.1, 0.9, self.n_tubes)
        self.surfaces = self.rhos ** 2

        # Link this object to the VMEC equilibrium
        super().__init__(depends_on=[vmec])

    def compute_tubes(self):
        """Extracts the features for all tubes on the uniform 75a grid."""
        self.vmec.run()
        nfp = self.vmec.wout.nfp
        s_half = self.vmec.s_half_grid
        iota_interp = interp1d(s_half, self.vmec.wout.iotas[1:], fill_value="extrapolate")

        centers = [
            (0.0, 0.0),
            (np.pi, 0.0),
            (0.0, np.pi / nfp),
            (np.pi, np.pi / nfp)
        ]

        # We will collect a specific scalar metric for each tube
        # (e.g., the variance of feature f3)
        tube_metrics = []

        for i in range(self.n_tubes):
            s_target = self.surfaces[i]
            # Cycle through the centers if we have more than 4 tubes
            theta_c, phi_c = centers[i % len(centers)]

            # 1. Get alpha
            iota_s = iota_interp(s_target)
            alpha_target = theta_c - iota_s * phi_c

            # 2. Track line
            aspect_ratio = self.vmec.aspect()
            target_length_in_minor_radii = 75.0

            # N_pol = (Target Length / Toroidal Circumference) * |iota|
            N_pol_exact = (target_length_in_minor_radii / (2 * np.pi * aspect_ratio)) * abs(iota_s)

            # Add a 50% safety factor to guarantee the line exceeds 75a,
            # and ensure a minimum of at least 1 turn.
            poloidal_turns = max(1.0, N_pol_exact * 1.5)
            theta1d = np.linspace(theta_c - poloidal_turns * np.pi,
                                  theta_c + poloidal_turns * np.pi, 3000)

            fl = vmec_fieldlines(self.vmec, s_target, alpha_target, theta1d=theta1d)

            # 3. Compute arclength (z)
            L_ref = fl.L_reference
            gradpar = fl.gradpar_theta_pest[0, 0, :]
            z_raw = cumulative_trapezoid(L_ref / gradpar, fl.theta_pest[0, 0, :], initial=0.0)

            # Shift z
            z_shifted = z_raw - z_raw[len(theta1d) // 2]

            # 4. Extract raw features
            s = s_target
            alpha = alpha_target
            psi_a = fl.edge_toroidal_flux_over_2pi
            sign_psi_a = fl.toroidal_flux_sign

            modB = fl.modB[0, 0, :]
            grad_s2 = fl.grad_s_dot_grad_s[0, 0, :]
            grad_alpha2 = fl.grad_alpha_dot_grad_alpha[0, 0, :]
            grad_alpha_dot_grad_psi = fl.grad_alpha_dot_grad_psi[0, 0, :]

            B_cross_grad_B_dot_grad_psi = fl.B_cross_grad_B_dot_grad_psi[0, 0, :]
            B_cross_grad_B_dot_grad_alpha = fl.B_cross_grad_B_dot_grad_alpha[0, 0, :]
            B_cross_kappa_dot_grad_psi = fl.B_cross_kappa_dot_grad_psi[0, 0, :]
            B_cross_kappa_dot_grad_alpha = fl.B_cross_kappa_dot_grad_alpha[0, 0, :]

            # assembling features
            f1_raw = modB

            f2_raw = (L_ref / (2 * np.sqrt(s) * psi_a)) * B_cross_grad_B_dot_grad_psi / (modB ** 3)

            f3_raw = -L_ref * sign_psi_a * (
                    (alpha / (2 * np.sqrt(s) * psi_a)) * B_cross_grad_B_dot_grad_psi +
                    np.sqrt(s) * B_cross_grad_B_dot_grad_alpha
            ) / (modB ** 3)

            f4_raw = -L_ref * sign_psi_a * (
                    (alpha / (2 * np.sqrt(s) * psi_a)) * B_cross_kappa_dot_grad_psi +
                    np.sqrt(s) * B_cross_kappa_dot_grad_alpha
            ) / (modB ** 2)

            f5_raw = (L_ref ** 2 / (4 * s)) * grad_s2

            f6_raw = -L_ref ** 2 * sign_psi_a * (
                    (alpha / (4 * s)) * grad_s2 +
                    0.5 * grad_alpha_dot_grad_psi / psi_a
            )

            f7_raw = L_ref ** 2 * (
                    (alpha ** 2 / (4 * s)) * grad_s2 +
                    (alpha / psi_a) * grad_alpha_dot_grad_psi +
                    s * grad_alpha2
            )

            # 5. Interpolate onto uniform 75*a grid
            target_length = 75.0 * L_ref
            z_uniform = np.linspace(-target_length / 2, target_length / 2, 500)

            f1_final = interp1d(z_shifted, f1_raw, kind='cubic')(z_uniform)
            f2_final = interp1d(z_shifted, f2_raw, kind='cubic')(z_uniform)
            f3_final = interp1d(z_shifted, f3_raw, kind='cubic')(z_uniform)
            f4_final = interp1d(z_shifted, f4_raw, kind='cubic')(z_uniform)
            f5_final = interp1d(z_shifted, f5_raw, kind='cubic')(z_uniform)
            f6_final = interp1d(z_shifted, f6_raw, kind='cubic')(z_uniform)
            f7_final = interp1d(z_shifted, f7_raw, kind='cubic')(z_uniform)

            target_1 = np.var(sigmoid((f1_final ** 2) * f4_final) * f5_final / f1_final)
            target_2 = np.mean(sigmoid((f1_final ** 2) * f4_final) * (f5_final ** 4) / (f1_final ** 2))
            target_3 = np.mean(sigmoid((f1_final ** 2) * f4_final) * (f5_final ** 2) / f1_final)
            target_4 = np.var(sigmoid((f1_final ** 2) * f4_final) * (f5_final ** 2) / f1_final)
            target_5 = np.mean(sigmoid((f1_final ** 2) * f4_final) * (f5_final ** 2) / (f1_final ** 2))
            tube_metrics.append(target_1)

        return tube_metrics

    def J(self):
        metrics = self.compute_tubes()
        return np.array(metrics)


mpi = MpiPartition(ngroups=4)

max_mode = 1
v = Vmec('input.LandremanPaul2021_QA_lowres', mpi)
surf = v.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")  # Major radius

gk_multi_obj = MultiTubeGyrokineticObjective(v, n_tubes=4)
prob = LeastSquaresProblem.from_sigma([0.0, 6.0], [1.0, 0.1], [gk_multi_obj.J, v.aspect])

print(f"Initial Multi-Tube Metric: {gk_multi_obj.J()}")
least_squares_mpi_solve(prob, mpi, grad=True)
print(f"Final Multi-Tube Metric: {gk_multi_obj.J()}")
