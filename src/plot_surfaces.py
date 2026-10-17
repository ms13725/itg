#!/usr/bin/env python
"""
Plots from wout_final.nc for one optimization run.

Usage: plot_surfaces.py <target>
  plot3.pdf    flux-surface cross sections at 4 toroidal angles
  heatmap.pdf  turbulence target <target> (1-5) on the 3D boundary: outer-surface value and radial average
"""

import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import RectBivariateSpline
from scipy.io import netcdf_file

from make_paraview import compute_target_fields
from wout_info import nfp_label, target_label

WOUT_FILE = 'wout_final.nc'

# How far to zoom into each 3D heatmap panel (1 = matplotlib default)
HEATMAP_ZOOM = 1.3

# Heatmap grid is upsampled this many times in theta and phi before plotting, so faces are smaller than pixels
HEATMAP_REFINE = 3


def read_wout(path):
    """Read the boundary/axis Fourier data needed for the plots. Asymmetric terms are zero when lasym is off."""
    with netcdf_file(path, 'r', mmap=False) as f:
        def read(name):
            return f.variables[name][()].copy()

        w = SimpleNamespace(
            nfp=int(read('nfp')), ns=int(read('ns')), ntor=int(read('ntor')),
            xm=read('xm'), xn=read('xn'),
            rmnc=read('rmnc'), zmns=read('zmns'),
            raxis_cc=read('raxis_cc'), zaxis_cs=read('zaxis_cs'),
        )
        if read('lasym__logical__') == 1:
            w.rmns, w.zmnc = read('rmns'), read('zmnc')
            w.raxis_cs, w.zaxis_cc = read('raxis_cs'), read('zaxis_cc')
        else:
            w.rmns, w.zmnc = np.zeros_like(w.rmnc), np.zeros_like(w.zmns)
            w.raxis_cs, w.zaxis_cc = np.zeros_like(w.raxis_cc), np.zeros_like(w.zaxis_cs)
    return w


def surface_rz(w, iradius, theta, phi):
    """R, Z of flux surface `iradius` on a (theta, phi) grid, shape (ntheta, nphi)."""
    angle = w.xm[:, None, None] * theta[None, :, None] - w.xn[:, None, None] * phi[None, None, :]
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.tensordot(w.rmnc[iradius], cos, axes=1) + np.tensordot(w.rmns[iradius], sin, axes=1)
    Z = np.tensordot(w.zmns[iradius], sin, axes=1) + np.tensordot(w.zmnc[iradius], cos, axes=1)
    return R, Z


def axis_rz(w, phi):
    """R, Z of the magnetic axis at each toroidal angle in phi."""
    angle = -np.arange(w.ntor + 1)[:, None] * w.nfp * phi[None, :]
    R = w.raxis_cc @ np.cos(angle) + w.raxis_cs @ np.sin(angle)
    Z = w.zaxis_cs @ np.sin(angle) + w.zaxis_cc @ np.cos(angle)
    return R, Z


def plot_cross_sections(w, filename='plot3.pdf'):
    """Boundary at 4 toroidal angles overlaid, then 10 nested surfaces + axis at each angle."""
    theta = np.linspace(0, 2 * np.pi, 200)
    phi = np.linspace(0, 2 * np.pi / w.nfp, 4, endpoint=False)
    iradii = np.linspace(0, w.ns - 1, 10).round().astype(int)
    R_axis, Z_axis = axis_rz(w, phi)

    legend_labels = [r'$\phi=0$', '1/4 period', '1/2 period', '3/4 period']
    titles = [r'$\phi=0$', r'1/4 period: $\phi=(1/4)2\pi/N_{fp}$', r'1/2 period: $\phi=(1/2)2\pi/N_{fp}$',
              r'3/4 period: $\phi=(3/4)2\pi/N_{fp}$']

    fig, axes = plt.subplots(2, 3, figsize=(9, 7))
    axes = axes.flat

    # Panel 1: boundary at each angle; legend sits above the axes so it doesn't cover the curves
    R, Z = surface_rz(w, w.ns - 1, theta, phi)
    ax = axes[0]
    for iphi, label in enumerate(legend_labels):
        ax.plot(R[:, iphi], Z[:, iphi], '-', label=label)
    ax.legend(fontsize='x-small', loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)

    # Panels 2-5: nested surfaces at each angle
    surfaces = [surface_rz(w, i, theta, phi) for i in iradii]
    for iphi, title in enumerate(titles):
        ax = axes[iphi + 1]
        for R, Z in surfaces:
            ax.plot(R[:, iphi], Z[:, iphi], '-')
        ax.plot(R_axis[iphi], Z_axis[iphi], 'xr')
        ax.set_title(title)

    for ax in axes[:5]:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('R [meters]')
        ax.set_ylabel('Z [meters]')
    axes[5].set_visible(False)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def refine(a, factor=HEATMAP_REFINE):
    """Cubic-spline upsample a (ntheta, nphi) array on a uniform grid by `factor` in each direction."""
    ntheta, nphi = a.shape
    u, v = np.linspace(0, 1, ntheta), np.linspace(0, 1, nphi)
    spline = RectBivariateSpline(u, v, a)
    return spline(np.linspace(0, 1, factor * ntheta), np.linspace(0, 1, factor * nphi))


def plot_target_heatmap(target, nfp, filename='heatmap.pdf'):
    """Target `target` painted on the 3D boundary: value at s = 1 (left) and averaged over s (right)."""
    surf, fields = compute_target_fields(WOUT_FILE, ntheta=64, nphi=150)
    xyz = surf.gamma()  # (nphi, ntheta, 3); fields are (ntheta, nphi)
    X, Y, Z = (refine(xyz[:, :, i].T) for i in range(3))
    extents = (np.ptp(X), np.ptp(Y), np.ptp(Z))

    note = '\n(squared deviation from surface mean)' if target in (1, 4) else ''
    panels = [(fields[f't{target}_outer'], f'Target {target} on boundary ($s = 1$)'),
              (fields[f't{target}_avg'], f'Target {target} averaged over $s = 0.1$–$1.0$')]

    fig = plt.figure(figsize=(10, 6))
    for ipanel, (field, title) in enumerate(panels):
        left = 0.5 * ipanel
        # Color range from the computed values, so spline overshoot doesn't stretch the colorbar
        norm = Normalize(vmin=field.min(), vmax=field.max())
        field = refine(field)

        # Each panel gets half the width and most of the height; colorbar goes underneath
        ax = fig.add_axes([left, 0.15, 0.5, 0.75], projection='3d')
        ax.plot_surface(X, Y, Z, facecolors=cm.jet(norm(field)), rstride=1, cstride=1, antialiased=False,
                        rasterized=True)  # keeps the PDF small with this many faces
        ax.set_box_aspect(extents, zoom=HEATMAP_ZOOM)  # true proportions
        ax.set_axis_off()
        ax.set_title(title + note)

        cax = fig.add_axes([left + 0.1, 0.08, 0.3, 0.03])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.jet), cax=cax, orientation='horizontal')

    fig.suptitle(f'{nfp_label(nfp)}')
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: plot_surfaces.py <target>')
    target = int(sys.argv[1])
    if target not in (1, 2, 3, 4, 5):
        sys.exit(f'target must be one of 1-5, got {target}')

    w = read_wout(WOUT_FILE)
    plot_cross_sections(w)
    plot_target_heatmap(target, w.nfp)


if __name__ == '__main__':
    main()
