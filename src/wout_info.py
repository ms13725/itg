import numpy as np
from matplotlib.lines import Line2D
from scipy.io import netcdf_file  # same reader simsopt's Vmec uses for wout files

# Marker for points whose value was clipped to the plot cutoff
CLIPPED_STYLE = dict(marker='^', color='red', linestyle='none', markersize=8, zorder=3)


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


def plot_clipped(ax, x, y, cutoff, **kwargs):
    """Plot y vs x with values above cutoff drawn at the cutoff and marked by a red triangle."""
    x = np.asarray(x)
    y = np.asarray(y)
    clipped = y > cutoff
    ax.plot(x, np.minimum(y, cutoff), **kwargs)
    ax.plot(x[clipped], np.full(clipped.sum(), cutoff), **CLIPPED_STYLE)


def add_clipped_legend(ax, cutoff, **legend_kwargs):
    """Draw ax's legend with an extra entry explaining the clipped-value marker."""
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], **CLIPPED_STYLE))
    labels.append(f"clipped (> {cutoff:g})")
    ax.legend(handles, labels, **legend_kwargs)
