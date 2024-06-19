import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from nitransforms.nonlinear import DenseFieldTransform

def plot_grid(path_to_file):
   return

def map_coords(xfm, index):
        """Calculate vector components of the field using the reference coordinates"""
        x = xfm.reference.ndcoords[0][0::index]
        y = xfm.reference.ndcoords[1][0::index]
        z = xfm.reference.ndcoords[2][0::index]

        u = xfm._field[...,0].flatten()[0::index] - x
        v = xfm._field[...,1].flatten()[0::index] - y
        w = xfm._field[...,2].flatten()[0::index] - z
        return x, y, z, u, v, w

def get_2dcenters(x, y, ind):
    samples_x = np.arange(x.min(), x.max(), step=(ind)**(1/3)).astype(int)
    samples_y = np.arange(y.min(), y.max(), step=(ind)**(1/3)).astype(int)

    lenx = len(samples_x)
    leny = len(samples_y)
    return zip(*product(samples_x, samples_y)), lenx, leny

path_to_file=Path("./tests/data/ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz")

fig = plt.figure()

xfm = DenseFieldTransform(
    path_to_file, 
    is_deltas=True
)

index = 100

x, y, z, u, v, w = map_coords(xfm, index)

gc, lenx, leny = get_2dcenters(x, y, index)
xy = list(gc)
lenxy = lenx * leny

x_moved, y_moved = [], []
#rounding len(u)/lenxy DOWN to ensure n_pts_moved > n_pts_grid
u = u[0::int(len(u)/lenxy)]
v = v[0::int(len(v)/lenxy)]

for ind, (i, j) in enumerate(zip(u, v)):
    try:
        x_moved.append(xy[0][ind] + i)
        y_moved.append(xy[1][ind] + j)
    except IndexError:
        break

# Plot grid
for ind, i in enumerate(x_moved):
    if ind%leny==0:
        plt.plot(x_moved[ind:leny+ind], y_moved[ind:leny+ind], c='k')
    plt.plot(x_moved[ind::leny], y_moved[ind::leny], c='k')

#plt.scatter(xy[0], xy[1], c='k')
#plt.scatter(x_moved, y_moved[0:len(x_moved)], s=5, c='r')
plt.show()
