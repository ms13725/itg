from scipy.io import netcdf_file  # same reader simsopt's Vmec uses for wout files


def read_nfp(wout_file="wout_final.nc"):
    """Return nfp from a VMEC wout file, or None if it can't be read."""
    try:
        with netcdf_file(wout_file, "r", mmap=False) as f:
            return int(f.variables["nfp"][()])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def nfp_label(nfp):
    return f"nfp = {nfp}" if nfp is not None else "nfp = ?"


def target_label(target):
    return f"turbulence target {target}"
