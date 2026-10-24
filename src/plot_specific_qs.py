import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys
import argparse
from fractions import Fraction
from wout_info import read_nfp, nfp_label, target_label, plot_clipped, add_clipped_legend

# Values above these are drawn at the cutoff as red triangles
ASPECT_CUTOFF = 20
TURB_CUTOFF = 10
QS_CUTOFF = 10


def plot_combined_residuals(aspect_target, iota_target, target, directory="."):
    # 1. Search for all residuals files in the specified directory
    file_pattern = os.path.join(directory, "residuals_*.dat")
    filepaths = glob.glob(file_pattern)

    # 2. Sort files chronologically
    filepaths.sort()

    if not filepaths:
        print(f"No files matching 'residuals_*.dat' found in {directory}", file=sys.stderr, flush=True)
        return

    print("Found and processing the following files in chronological order:")
    for fp in filepaths:
        print(f" - {os.path.basename(fp)}")

    # 3. Initialize the subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    total_evals = 0

    # 4. Loop through the sorted files and plot
    for i, filepath in enumerate(filepaths):
        # Read and filter the data
        df = pd.read_csv(filepath, skiprows=4, skipinitialspace=True)
        valid_data = df[df['F(0)'] < 1e10].copy()

        # Shift the x-axis for continuity
        valid_data['adjusted_eval'] = valid_data['function_evaluation'] + total_evals

        # Reverse the least-squares scaling for scalar targets using input targets
        valid_data['aspect_ratio_actual'] = valid_data['F(0)'] + aspect_target
        valid_data['mean_iota_actual'] = valid_data['F(1)'] + iota_target

        # Residual layout follows the order of terms in run_optimization.py:
        #   F(0)            aspect
        #   F(1)            mean iota
        #   F(2)..F(N-5)    quasisymmetry residuals (one per surface/theta/phi grid point)
        #   last 4 F cols   turbulence proxy (turb_obj.J)
        f_columns = [c for c in df.columns if c.startswith('F(')]
        n_turb = 4
        qs_columns = f_columns[2:-n_turb]
        turb_columns = f_columns[-n_turb:]

        valid_data['turbulence_norm'] = np.linalg.norm(valid_data[turb_columns].values, axis=1)
        # Same as QuasisymmetryRatioResidual.total(): sum of squared residuals
        valid_data['qs_total'] = np.sum(valid_data[qs_columns].values ** 2, axis=1)

        # Plot Subplot 1: Aspect Ratio
        plot_clipped(
            axs[0],
            valid_data['adjusted_eval'],
            valid_data['aspect_ratio_actual'],
            ASPECT_CUTOFF,
            marker='o',
            linestyle='-',
            label=f'loop iteration = {i + 1}'
        )

        # Plot Subplot 2: Mean Iota
        axs[1].plot(
            valid_data['adjusted_eval'],
            valid_data['mean_iota_actual'],
            marker='o',
            linestyle='-'
        )

        # Plot Subplot 3: Geometric Turbulence components
        plot_clipped(
            axs[2],
            valid_data['adjusted_eval'],
            valid_data['turbulence_norm'],
            TURB_CUTOFF,
            marker='o',
            linestyle='-'
        )

        # Plot Subplot 4: Quasisymmetry
        plot_clipped(
            axs[3],
            valid_data['adjusted_eval'],
            valid_data['qs_total'],
            QS_CUTOFF,
            marker='o',
            linestyle='-'
        )

        # Draw vertical lines for step boundaries
        if i > 0:
            step_start = valid_data['adjusted_eval'].min()
            for ax in axs:
                ax.axvline(x=step_start, color='gray', linestyle='--', alpha=0.7)

        # Update the running total of evaluations
        if not valid_data.empty:
            total_evals += valid_data['function_evaluation'].max() + 1

    # 5. Format the Aspect Ratio Subplot
    axs[0].axhline(y=aspect_target, color='r', linestyle='--', label=f'Target ({aspect_target})')
    axs[0].set_ylabel('Aspect Ratio')
    axs[0].set_title('Aspect Ratio Convergence')
    add_clipped_legend(axs[0], ASPECT_CUTOFF)
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # 6. Format the Mean Iota Subplot
    axs[1].axhline(y=iota_target, color='r', linestyle='--', label=f'Target ({iota_target})')
    axs[1].set_ylabel('Mean Iota')
    axs[1].set_title('Mean Iota Convergence')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # 7. Format the Geometric Turbulence Subplot (target is 0, so use a log axis)
    axs[2].set_yscale('log')
    axs[2].set_ylabel('Turbulence Proxy Norm')
    axs[2].set_title('Turbulence Proxy Norm Convergence')
    add_clipped_legend(axs[2], TURB_CUTOFF)
    axs[2].grid(True, which='both', linestyle='--', alpha=0.6)

    # 8. Format the Quasisymmetry Subplot (target is 0, so use a log axis)
    axs[3].set_yscale('log')
    axs[3].set_ylabel('QS Objective (sum of squares)')
    axs[3].set_title('Quasisymmetry Convergence')
    axs[3].set_xlabel('Cumulative Function Evaluations')
    add_clipped_legend(axs[3], QS_CUTOFF)
    axs[3].grid(True, which='both', linestyle='--', alpha=0.6)

    # Label the figure with the device's field-period count
    nfp = read_nfp(os.path.join(directory, "wout_final.nc"))
    fig.suptitle(f"{nfp_label(nfp)}  —  {target_label(target)}")

    # Display the final composite plot
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('targets.pdf')
    plt.close(fig)


def helicity_label(helicity_m, helicity_n):
    """Name the quasisymmetry type targeted by (M, N)."""
    if helicity_n == 0:
        kind = "QA"
    elif helicity_m == 0:
        kind = "QP"
    else:
        kind = "QH"
    return f"{kind} (M={helicity_m}, N={helicity_n})"


def pi_label(frac):
    """Format Fraction(p, q) as a tick label for p*pi/q."""
    p, q = frac.numerator, frac.denominator
    if p == 0:
        return "0"
    num = "π" if p == 1 else f"{p}π"
    return num if q == 1 else f"{num}/{q}"


def plot_boozer(wout_file, target, helicity_m, helicity_n, mboz=48, nboz=48, outfile="boozer.pdf"):
    """
    Transform the VMEC equilibrium to Boozer coordinates on the surfaces
    targeted by QuasisymmetryRatioResidual in run_optimization.py and plot
    |B| contours over one field period on each.
    """
    try:
        import booz_xform as bx
    except ImportError:
        print("booz_xform is not installed; skipping Boozer plot", file=sys.stderr, flush=True)
        return

    print(f"making Boozer plot from {wout_file}")
    b = bx.Booz_xform()
    b.verbose = 0
    b.read_wout(wout_file)
    b.mboz = mboz
    b.nboz = nboz

    # Same surfaces as the QS objective; snap each to the nearest VMEC half-grid surface
    s_targets = np.linspace(0.2, 0.8, 4)
    s_in = np.asarray(b.s_in)
    js_list = [int(np.argmin(np.abs(s_in - s))) for s in s_targets]
    unique_js = sorted(set(js_list))
    b.compute_surfs = unique_js
    b.run()

    nfp = b.nfp
    xm = np.asarray(b.xm_b)[:, None, None]
    xn = np.asarray(b.xn_b)[:, None, None]

    # One field period in phi, full poloidal turn in theta
    theta_1d = np.linspace(0, 2 * np.pi, 64)
    phi_1d = np.linspace(0, 2 * np.pi / nfp, 64)
    phi_2d, theta_2d = np.meshgrid(phi_1d, theta_1d)
    angle = xm * theta_2d[None, :, :] - xn * phi_2d[None, :, :]
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle) if b.asym else None

    # Tick positions/labels: quarter field periods in phi, quarter turns in theta
    phi_fracs = [Fraction(2 * k, 4 * nfp) for k in range(5)]
    theta_fracs = [Fraction(2 * k, 4) for k in range(5)]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    axs = axs.ravel()

    for ax, s_nominal, js in zip(axs, s_targets, js_list):
        # column of the booz_xform output arrays holding this surface
        j = unique_js.index(js)
        modB = np.einsum('k,kij->ij', np.asarray(b.bmnc_b)[:, j], cos_angle)
        if b.asym:
            modB += np.einsum('k,kij->ij', np.asarray(b.bmns_b)[:, j], sin_angle)

        cs = ax.contour(phi_2d, theta_2d, modB, levels=25, cmap='viridis')
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label('|B| [T]')

        ax.set_title(f"s = {s_nominal:g}")
        ax.set_xticks([float(f) * np.pi for f in phi_fracs])
        ax.set_xticklabels([pi_label(f) for f in phi_fracs])
        ax.set_yticks([float(f) * np.pi for f in theta_fracs])
        ax.set_yticklabels([pi_label(f) for f in theta_fracs])

    for ax in axs[2:]:
        ax.set_xlabel(r'Boozer toroidal angle $\varphi_B$ [rad]')
    for ax in axs[::2]:
        ax.set_ylabel(r'Boozer poloidal angle $\theta_B$ [rad]')

    fig.suptitle(f"Boozer |B|  —  {nfp_label(nfp)}  —  {helicity_label(helicity_m, helicity_n)}"
                 f"  —  {target_label(target)}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outfile)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot convergence of SIMSOPT optimization residuals (with QS).")
    parser.add_argument("aspect_target", type=float, help="Target aspect ratio")
    parser.add_argument("iota_target", type=float, help="Target mean iota")
    parser.add_argument("target", type=int, help="Turbulence target (1-5) used in the optimization")
    parser.add_argument("helicity_m", type=int, help="QS helicity m used in the optimization")
    parser.add_argument("helicity_n", type=int, help="QS helicity n used in the optimization")
    parser.add_argument("--directory", type=str, default=".", help="Directory containing residuals files")
    parser.add_argument("--mboz", type=int, default=48, help="Boozer poloidal resolution")
    parser.add_argument("--nboz", type=int, default=48, help="Boozer toroidal resolution")

    args = parser.parse_args()
    print('making residual plots')
    plot_combined_residuals(args.aspect_target, args.iota_target, args.target, directory=args.directory)

    wout_file = os.path.join(args.directory, "wout_final.nc")
    plot_boozer(wout_file, args.target, args.helicity_m, args.helicity_n, mboz=args.mboz, nboz=args.nboz)
