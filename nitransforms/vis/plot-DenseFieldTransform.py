import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

import nibabel as nb
from nitransforms.linear import Affine
from nitransforms.nonlinear import DenseFieldTransform

index = 10000

rand_field = np.random.rand(57, 67, 56, 3)
Nifti_img = nb.load("../tests/data/ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz")
nii_data = Nifti_img.get_fdata()
nii_aff  = Nifti_img.affine
nii_hdr  = Nifti_img.header

xfm = DenseFieldTransform(
        nii_data,
        is_deltas=False,
        reference=nb.Nifti1Image(nii_data, np.eye(4), None),
    )

"""Calculate vector components of the field using the reference coordinates"""
i = xfm.reference.ndcoords[0]
j = xfm.reference.ndcoords[1]
k = xfm.reference.ndcoords[2]

u = xfm._field[...,0].flatten() - i
v = xfm._field[...,1].flatten() - j
w = xfm._field[...,2].flatten() - k

magnitude = np.sqrt(u**2 + v**2 + w**2)

clr_xy = np.hypot(u, v)[0::index]/np.hypot(u, v)[0::index].max()
clr_xz = np.hypot(u, w)[0::index]/np.hypot(u, w)[0::index].max()
clr3d = plt.cm.viridis(magnitude[0::index]/magnitude[0::index].max())

"""Plot"""
fig = plt.figure(figsize=(12,10)) #(12, 6) for gs(2,3)
#gs = GridSpec(2, 2, figure=fig, wspace=1/4, hspace=1/3)
gs = GridSpec(2, 3, figure=fig, wspace=1/4, hspace=1/2.5)

fig.suptitle(str("Non-Linear DenseFieldTransform field \n (DataSource: nii_data, elements: [0::"+str(index)+"])"), fontsize='14')

ax1 = fig.add_subplot(gs[0,0])
q1 = ax1.quiver(i[0::index], j[0::index], u[0::index], v[0::index], clr_xy, cmap='viridis')
ax1.set_title("i-j projection", weight='bold')
ax1.set_xlabel("i")
ax1.set_ylabel("j")
#plt.colorbar(q1)

ax3 = fig.add_subplot(gs[1,0])
q3 = ax3.quiver(i[0::index], k[0::index], u[0::index], w[0::index], clr_xz, cmap='viridis')
ax3.set_title("i-k projection", weight='bold')
ax3.set_xlabel("i")
ax3.set_ylabel("k")
#plt.colorbar(q3)

ax2 = fig.add_subplot(gs[:,1:], projection='3d')
q2 = ax2.quiver(i[0::index], j[0::index], k[0::index], u[0::index], v[0::index], w[0::index], colors=clr3d, length=0.1)
ax2.set_title("3D projection", weight='bold')
ax2.set_xlabel("x"); ax2.xaxis.set_rotate_label(False)
ax2.set_ylabel("y"); ax2.yaxis.set_rotate_label(False)
ax2.set_zlabel("z"); ax2.zaxis.set_rotate_label(False)
#plt.colorbar(q2, location='bottom')#, shrink=1, aspect=30)
#plt.colorbar(q1, ax=[ax1,ax3], location='right', shrink=0.6, aspect=50)

fig.colorbar(q1, ax=[ax1, ax3, ax2], cmap='viridis', location='bottom', shrink=0.8, aspect=50)
plt.savefig("/Users/julienmarabotto/workspace/nonlinear-field-index-"+str(index)+"-niftidata.jpg", dpi=300)
plt.show()


