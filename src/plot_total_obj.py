import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys
import argparse
from wout_info import read_nfp, nfp_label, target_label


def plot_combined_simsopt_objectives(target, directory="."):
    # 1. Search for all objective files in the specified directory
    file_pattern = os.path.join(directory, "objective_*.dat")
    filepaths = glob.glob(file_pattern)

    # 2. Sort files chronologically
    # (Alphabetical sorting works perfectly for YYYY-MM-DD-HH-MM-SS timestamps)
    filepaths.sort()

    # 3. Check if any files were found before proceeding
    if not filepaths:
        print(f"No files matching 'objective_*.dat' found in {directory}", file=sys.stderr, flush=True)
        return

    print("Found and processing the following files in chronological order:")
    for fp in filepaths:
        print(f" - {os.path.basename(fp)}")

    # 4. Initialize the plot
    plt.figure(figsize=(10, 6))
    total_evals = 0

    # 5. Loop through the sorted files and plot
    for i, filepath in enumerate(filepaths):
        # Read and filter the data
        df = pd.read_csv(filepath, skiprows=4, skipinitialspace=True)
        valid_data = df[df['objective_function'] < 1e20].copy()

        # Shift the x-axis so the new step starts where the last one ended
        valid_data['adjusted_eval'] = valid_data['function_evaluation'] + total_evals

        # Plot this specific continuation step
        plt.plot(
            valid_data['adjusted_eval'],
            valid_data['objective_function'],
            marker='o',
            linestyle='-',
            label=f'loop iteration = {i + 1}'
        )

        # Update the running total of evaluations
        if not valid_data.empty:
            total_evals += valid_data['function_evaluation'].max() + 1

        # Draw a vertical dashed line to mark the transition
        if i > 0:
            plt.axvline(x=valid_data['adjusted_eval'].min(), color='gray', linestyle='--', alpha=0.7)

    # 6. Format and display
    plt.yscale('log')
    plt.xlabel('Cumulative Function Evaluations')
    plt.ylabel('Total Objective Function')
    nfp = read_nfp(os.path.join(directory, "wout_final.nc"))
    plt.title(f'Optimization Progress ({nfp_label(nfp)}, {target_label(target)})')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('total_obj.pdf')


# Run the function. It defaults to the current working directory (".").
# You can also pass a specific path: plot_combined_simsopt_objectives("/path/to/your/data")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot the total objective across continuation steps.")
    parser.add_argument("target", type=int, help="Turbulence target (1-5) used in the optimization")
    args = parser.parse_args()
    print('making total_obj plot')
    plot_combined_simsopt_objectives(args.target)