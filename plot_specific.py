import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys
import argparse


def plot_combined_residuals(aspect_target, iota_target, directory="."):
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
        axs[0].plot(
            valid_data['adjusted_eval'],
            valid_data['aspect_ratio_actual'],
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
        axs[2].plot(
            valid_data['adjusted_eval'],
            valid_data['turbulence_norm'],
            marker='o',
            linestyle='-'
        )

        # Plot Subplot 4: Quasisymmetry
        axs[3].plot(
            valid_data['adjusted_eval'],
            valid_data['qs_total'],
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
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # 6. Format the Mean Iota Subplot
    axs[1].axhline(y=iota_target, color='r', linestyle='--', label=f'Target ({iota_target})')
    axs[1].set_ylabel('Mean Iota')
    axs[1].set_title('Mean Iota Convergence')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # 7. Format the Geometric Turbulence Subplot
    axs[2].axhline(y=0.0, color='r', linestyle='--', label=f'Target (0.0)')
    axs[2].set_ylabel('Turbulence Proxy Norm')
    axs[2].set_title('Turbulence Proxy Norm Convergence')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.6)

    # 8. Format the Quasisymmetry Subplot (target is 0, so use a log axis)
    axs[3].set_yscale('log')
    axs[3].set_ylabel('QS Objective (sum of squares)')
    axs[3].set_title('Quasisymmetry Convergence')
    axs[3].set_xlabel('Cumulative Function Evaluations')
    axs[3].grid(True, which='both', linestyle='--', alpha=0.6)

    # Display the final composite plot
    plt.tight_layout()
    plt.savefig('targets.pdf')
    plt.close(fig)


def plot_boozer(wout_file, s_target=0.5, mboz=48, nboz=48, outfile="boozer.pdf"):
    """
    Transform the VMEC equilibrium to Boozer coordinates on the surface
    closest to s_target and plot |B| contours over one field period.
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

    # Only transform the one surface we need (s_in is VMEC's half grid)
    js_in = int(np.argmin(np.abs(np.asarray(b.s_in) - s_target)))
    b.compute_surfs = [js_in]
    b.run()

    fig = plt.figure(figsize=(6, 4.5))
    bx.surfplot(b, js=0, fill=False, ncontours=25)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot convergence of SIMSOPT optimization residuals.")
    parser.add_argument("aspect_target", type=float, help="Target aspect ratio")
    parser.add_argument("iota_target", type=float, help="Target mean iota")
    parser.add_argument("--directory", type=str, default=".", help="Directory containing residuals files")
    parser.add_argument("--boozer-s", type=float, default=0.5, help="Surface s for the Boozer plot")
    parser.add_argument("--mboz", type=int, default=48, help="Boozer poloidal resolution")
    parser.add_argument("--nboz", type=int, default=48, help="Boozer toroidal resolution")

    args = parser.parse_args()
    print('making residual plots')
    plot_combined_residuals(args.aspect_target, args.iota_target, directory=args.directory)

    wout_file = os.path.join(args.directory, "wout_final.nc")
    plot_boozer(wout_file, s_target=args.boozer_s, mboz=args.mboz, nboz=args.nboz)