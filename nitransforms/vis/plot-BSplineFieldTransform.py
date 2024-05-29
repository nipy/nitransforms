import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import nibabel as nb
from nitransforms.linear import Affine
from nitransforms.nonlinear import BSplineFieldTransform

rand_field = np.random.rand(57, 67, 56, 3)
some_zeros = np.zeros((10, 10, 10, 3))

Nifti_img = nb.load("../tests/data/ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz")
nii_data = Nifti_img.get_fdata()
nii_aff  = Nifti_img.affine
nii_hdr  = Nifti_img.header
nii_ref =nb.Nifti1Image(nii_data, np.eye(4), None)

xfm = BSplineFieldTransform(
        coefficients = Nifti_img
    ).to_field(reference=nii_ref)

import pdb; pdb.set_trace()
"""Calculate vector components of the field using the reference coordinates"""
x = xfm.reference.ndcoords[0]
y = xfm.reference.ndcoords[1]
z = xfm.reference.ndcoords[2]

u = xfm._field[...,0].flatten() - x
v = xfm._field[...,1].flatten() - y
w = xfm._field[...,2].flatten() - z

clr_xy = np.hypot(u, v)
clr_xz = np.hypot(u, w)

"""Plot"""
index = 10000

fig = plt.figure(figsize=(10,8))
gs = GridSpec(2, 2, figure=fig, hspace=1/3)

fig.suptitle(str("Non-Linear DenseFieldTransform field (DataSource: nii_data, elements: [0::"+str(index)+"])"), fontsize='14')

ax1 = fig.add_subplot(gs[0,0])
q1 = ax1.quiver(x[0::index], y[0::index], u[0::index], v[0::index], clr_xy[0::index])
ax1.set_title("x-y projection", weight='bold')
ax1.set_xlabel("x")
ax1.set_ylabel("y")
plt.colorbar(q1)

ax3 = fig.add_subplot(gs[1,0])
q3 = ax3.quiver(x[0::index], z[0::index], u[0::index], w[0::index], clr_xz[0::index])
ax3.set_title("x-z projection", weight='bold')
ax3.set_xlabel("x")
ax3.set_ylabel("z")
plt.colorbar(q3)

ax2 = fig.add_subplot(gs[:,1], projection='3d')
ax2.quiver(x[0::index], y[0::index], z[0::index], u[0::index], v[0::index], w[0::index], length=10, normalize=True)
ax2.set_title("3D projection", weight='bold')
ax2.set_xlabel("x"); ax2.xaxis.set_rotate_label(False)
ax2.set_ylabel("y"); ax2.yaxis.set_rotate_label(False)
ax2.set_zlabel("z"); ax2.zaxis.set_rotate_label(False)

#plt.savefig("/Users/julienmarabotto/workspace/nonlinear-field-index-"+str(index)+"-niftidata.jpg")
plt.show()
exit()

"""Plotting"""
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(121)
ax1 = sns.heatmap(xfm_linear.matrix, cmap='viridis')
ax1.set_title(str(type(xfm_linear)))

ax2 = fig.add_subplot(122)
ax2 = sns.heatmap(xfm.map(xfm._field[0][0], inverse=False), cmap='viridis')
#ax2 = sns.heatmap(xfm.map([np.eye(4), np.eye(4)], inverse=False)[..., 0], cmap='viridis')
ax2.set_title(str(type(xfm)))

plt.show()
exit()

"""structure for quiverplot"""
u, v = np.meshgrid(x, y)
C = np.hypot(u, v)

fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(121)
q = ax1.quiver(x, y, u, v, C)

plt.colorbar(q)
plt.show()




