import argparse
import shutil
import numpy as np
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.util import MpiPartition
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from grid_geom_features import MultiSurfaceTurbulenceObjective

parser = argparse.ArgumentParser(
    description="Run VMEC optimization with target parameters."
)
parser.add_argument("vmec_filename", type=str, help="Path to the VMEC input file")
parser.add_argument("aspect_target", type=float, help="Target aspect ratio")
parser.add_argument("iota_target", type=float, help="Target mean iota")
parser.add_argument("--target", type=int, default=1, choices=[1, 2, 3, 4, 5],
                    help="Turbulence target from grid_geom_features (default: 1)")
parser.add_argument("--helicity-m", type=int, default=None,
                    help="QS helicity m; omit with --helicity-n to skip quasisymmetry")
parser.add_argument("--helicity-n", type=int, default=None,
                    help="QS helicity n; omit with --helicity-m to skip quasisymmetry")

args = parser.parse_args()

use_qs = args.helicity_m is not None or args.helicity_n is not None
if use_qs and (args.helicity_m is None or args.helicity_n is None):
    parser.error("--helicity-m and --helicity-n must be given together")

mpi = MpiPartition()

vmec = Vmec(args.vmec_filename, mpi)
surf = vmec.boundary

turb_obj = MultiSurfaceTurbulenceObjective(vmec, target=args.target)

terms = [
    (vmec.aspect, args.aspect_target, 5),
    (vmec.mean_iota, args.iota_target, 20),
]
if use_qs:
    qs = QuasisymmetryRatioResidual(vmec,
                                    np.linspace(0.2, 0.8, 4),  # adjust later if needed, currently matching grid_geom_features
                                    helicity_m=args.helicity_m, helicity_n=args.helicity_n)
    terms.append((qs.residuals, 0, 1))
# turbulence objective must always be the last term
terms.append((turb_obj.J, 0.0, 1))

prob = LeastSquaresProblem.from_tuples(terms)

if mpi.proc0_world:
    qs_msg = f"QS on (m={args.helicity_m}, n={args.helicity_n})" if use_qs else "QS off"
    print(f"turbulence target {args.target}, {qs_msg}")

for step in range(3):
    max_mode = step + 2
    vmec.indata.mpol = 2 + step
    vmec.indata.ntor = vmec.indata.mpol

    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")

    least_squares_mpi_solve(prob, mpi, grad=True, save_residuals=True, max_nfev=30)

# least_squares_mpi_solve leaves prob.x at the optimum but does not run
# VMEC there, so no wout file exists for the final point yet. Run it once
# more and save it under a fixed name for the plotting scripts.
vmec.run()
if mpi.proc0_world:
    shutil.copy(vmec.output_file, "wout_final.nc")
    print(f"final equilibrium saved as wout_final.nc (from {vmec.output_file})")
