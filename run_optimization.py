import argparse
from simsopt.mhd import Vmec
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
# parser.add_argument("turb_target", type=float, help="Target turbulence objective value")

args = parser.parse_args()

mpi = MpiPartition()

vmec = Vmec(args.vmec_filename, mpi)
surf = vmec.boundary

turb_obj = MultiSurfaceTurbulenceObjective(vmec)


prob = LeastSquaresProblem.from_tuples([
    (vmec.aspect, args.aspect_target, 1),
    (vmec.mean_iota, args.iota_target, 1),
    (turb_obj.J, 0.0, 1),
])

for step in range(3):
    max_mode = step + 2
    vmec.indata.mpol = 2 + step
    vmec.indata.ntor = vmec.indata.mpol

    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")

    least_squares_mpi_solve(prob, mpi, grad=True, save_residuals=True, max_nfev=3)
