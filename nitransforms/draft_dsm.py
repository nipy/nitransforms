import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from nitransforms.nonlinear import DenseFieldTransform

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

xfm = DenseFieldTransform(
    path_to_file, 
    is_deltas=True
)

index = 100

x, y, z, u, v, w = map_coords(xfm, index)

c_blues, c_reds, c_greens, zeros = [], [], [], []

for ind, (i, j, k) in enumerate(zip(x, y, z)):
    if np.abs(u[ind]) > [np.abs(v[ind]) and np.abs(w[ind])]:
        c_blues.append((i, j, k, u[ind]))
    elif np.abs(v[ind]) > [np.abs(u[ind]) and np.abs(w[ind])]:
        c_reds.append((i, j, k, v[ind]))
    elif np.abs(w[ind]) > [np.abs(u[ind]) and np.abs(v[ind])]:
        c_greens.append((i, j, k, w[ind]))
    else:
        zeros.append(0)

assert len(np.concatenate((c_blues, c_reds, c_greens))) == len(x) - len(zeros)

fig = plt.figure()

c_blues = np.asanyarray(c_blues)
c_reds = np.asanyarray(c_reds)
c_greens = np.asanyarray(c_greens)

plt.scatter(c_blues[:, 0], c_blues[:, 1], c=(c_blues[:, -1]), cmap='Blues', s=1, alpha=1)
plt.scatter(c_reds[:, 0], c_reds[:, 1], c=(c_reds[:, -1]), cmap='Reds', s=1, alpha=1)
plt.scatter(c_greens[:, 0], c_greens[:, 1], c=(c_greens[:, -1]), cmap='Greens', s=1, alpha=1)

plt.show()
exit()
 
a, b, c, d = c_blues[0]
print(a, c_blues[0][0]) #equivalent
print(b)
print(c)
print(d)