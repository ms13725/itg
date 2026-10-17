import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys
import argparse
from wout_info import read_nfp, nfp_label, target_label


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
    fig, axs = plt.subplots(3, 1, figsize=(10, 11.25), sharex=True)
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

        # Residual layout follows the order of terms in run_optimization.py (no QS term):
        #   F(0)            aspect
        #   F(1)            mean iota
        #   last 4 F cols   turbulence proxy (turb_obj.J)
        f_columns = [c for c in df.columns if c.startswith('F(')]
        n_turb = 4
        turb_columns = f_columns[-n_turb:]

        valid_data['turbulence_norm'] = np.linalg.norm(valid_data[turb_columns].values, axis=1)

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
    axs[2].set_xlabel('Cumulative Function Evaluations')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.6)

    # Label the figure with the device's field-period count
    nfp = read_nfp(os.path.join(directory, "wout_final.nc"))
    fig.suptitle(f"{nfp_label(nfp)}  —  {target_label(target)}")

    # Display the final composite plot
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('targets.pdf')
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot convergence of SIMSOPT optimization residuals (no QS).")
    parser.add_argument("aspect_target", type=float, help="Target aspect ratio")
    parser.add_argument("iota_target", type=float, help="Target mean iota")
    parser.add_argument("target", type=int, help="Turbulence target (1-5) used in the optimization")
    parser.add_argument("--directory", type=str, default=".", help="Directory containing residuals files")

    args = parser.parse_args()
    print('making residual plots')
    plot_combined_residuals(args.aspect_target, args.iota_target, args.target, directory=args.directory)
