import numpy as np
import json
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from grid_geom_features import MultiSurfaceGyrokineticObjective

mpi = MpiPartition()

if mpi.proc0_world:
    history = {
        'aspect': [],
        'iota': [],
        'gk_obj': []
    }


class LoggedMultiGkObj(MultiSurfaceGyrokineticObjective):
    def J(self):
        val = super().J()

        if mpi.proc0_world:
            history['aspect'].append(self.vmec.aspect())
            history['iota'].append(self.vmec.mean_iota())
            history['gk_obj'].append(float(np.linalg.norm(val)))

        return val


vmec = Vmec('input.LandremanPaul2021_QA_lowres', mpi)
surf = vmec.boundary

# Instantiate Logged subclass instead of the original
gk_multi_obj = LoggedMultiGkObj(vmec)

# Now gk_multi_obj.J is a bound method of an Optimizable object,
# so it has the __self__ attribute simsopt expects.
prob = LeastSquaresProblem.from_tuples([
    (vmec.aspect, 6, 1),
    (vmec.mean_iota, 0.42, 1),
    (gk_multi_obj.J, 0, 1),
])

for step in range(2):
    max_mode = step + 1
    vmec.indata.mpol = 2 + step
    vmec.indata.ntor = vmec.indata.mpol

    if mpi.proc0_world:
        print(f'Beginning optimization with max_mode = {max_mode}\n'
              f'vmec mpol=ntor = {vmec.indata.mpol}\n'
              f'Previous vmec iteration = {vmec.iter}')

    # Define parameter space
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")

    least_squares_mpi_solve(prob, mpi, grad=True, max_nfev=20)

    if mpi.proc0_world:
        print(f'Done optimization with max_mode = {max_mode}\n'
              f'Final vmec iteration = {vmec.iter}')

if mpi.proc0_world:
    output_filename = 'optimization_history.json'
    with open(output_filename, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"Data successfully serialized to {output_filename}")
