import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import nibabel as nb
from nitransforms.linear import Affine
from nitransforms.nonlinear import DenseFieldTransform

def read_nifti(image_path):
    Nifti_img = nb.load(image_path)
    nii_data = Nifti_img.get_fdata()
    nii_aff  = Nifti_img.affine
    nii_hdr  = Nifti_img.header
    return nii_data, nii_aff, nii_hdr

def format_fig(figsize, gs_rows, gs_cols, gs_wspace, gs_hspace):
    fig = plt.figure(figsize=figsize) #(12, 6) for gs(2,3)
    fig.suptitle(str("Non-Linear DenseFieldTransform field"), fontsize='20', weight='bold')
    gs = GridSpec(gs_rows, gs_cols, figure=fig, wspace=gs_wspace, hspace=gs_hspace)
    return fig, gs

def format_axes(axis, title, xlabel, ylabel, zlabel, xticks, yticks, zticks):
    '''Format the figure axes. For 2D plots, zlabel and zticks parameters are None.'''
    
    axis.set_title(title, weight='bold')
    axis.set_xlabel(xlabel, fontsize=16)
    axis.set_ylabel(ylabel, fontsize=16)
    axis.set_xticks((xticks))
    axis.set_yticks((yticks))

    '''if 3d projection plot'''
    try:
        axis.set_zlabel(zlabel, fontsize=16)
        axis.set_zticks((zticks))
        axis.xaxis.set_rotate_label(False)
        axis.yaxis.set_rotate_label(False)
        axis.zaxis.set_rotate_label(False)

        #axis.plot(np.zeros(len(k)), np.zeros(len(k)), k, color='red', alpha=0.2)
    except:
        pass
   
    axis.tick_params(labelsize=14)
    #axis.plot(np.zeros(len(j)), j, color='red', alpha=0.2)
    #axis.plot(i, np.zeros(len(i)), color='red', alpha=0.2)
    return


savepath = str("/Users/julienmarabotto/workspace/Neuroimaging/plots/quiver/")
index = 10000

"""Read nifti image"""
nii_data, nii_aff, nii_hdr = read_nifti("../tests/data/ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz")
eye_aff = np.eye(4)
eg_aff = [[3,0,0,-78], [0,3*np.cos(0.3),-np.sin(0.3),-76], [0, np.sin(0.3), 3*np.cos(0.3), -64], [0,0,0,1]]

"""Define nonlinear transform"""
xfm = DenseFieldTransform(
        "../tests/data/ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz",
        is_deltas=True,
    )

"""Calculate vector components of the field using the reference coordinates"""
i = xfm.reference.ndcoords[0]
j = xfm.reference.ndcoords[1]
k = xfm.reference.ndcoords[2]

u = xfm._field[...,0].flatten() - i
v = xfm._field[...,1].flatten() - j
w = xfm._field[...,2].flatten() - k

magnitude = np.sqrt(u**2 + v**2 + w**2)

clr_xy = np.hypot(u, v)[0::index]
clr_xz = np.hypot(u, w)[0::index]
clr3d = plt.cm.viridis(magnitude[0::index]/magnitude[0::index].max())

i_max, i_min = i[0::index].max(), i[0::index].min()
j_max, j_min = j[0::index].max(), j[0::index].min()
k_max, k_min = k[0::index].max(), k[0::index].min()

"""Plot"""
fig, gs = format_fig(figsize=(15, 8), gs_rows=2, gs_cols=3, gs_wspace=1/4, gs_hspace=1/2.5)

ax1 = fig.add_subplot(gs[0,0])
ax_params = format_axes(ax1, "x-y projection", "x", "y", None, [-250, -200, -150, -100, -50, 0], [-275, -250, -200, -150, -100, -50, 0], None)
ax1.add_patch(mpl.patches.Rectangle((-150,-260), 75, 60, edgecolor='r', linewidth=1, facecolor='none'))
q1 = ax1.quiver(i[0::index], j[0::index], u[0::index], v[0::index], clr_xy, cmap='viridis', angles='xy', scale_units='xy', scale=1)
plt.colorbar(q1)

ax3 = fig.add_subplot(gs[1,0])
ax_params = format_axes(ax3, "x-y zoom", "x", "y", None, [-75, -100, -125, -150], [-260, -240, -220, -200, -180], None)
ax3.set_xlim(-150, -75)
ax3.set_ylim(-260, -180)
q3 = ax3.quiver(i[0::index], j[0::index], u[0::index], v[0::index], clr_xy, cmap='viridis', angles='xy', scale_units='xy', scale=1)
plt.colorbar(q3)

"""
ax_params = format_axes(ax3, "i-k zoom", "i", "k", None, [-75, -100, -125, -150, -175], [k_ticks], None)
q3 = ax3.quiver(i[0::index], k[0::index], u[0::index], w[0::index], clr_xz, cmap='viridis')
#plt.colorbar(q3)
"""

ax2 = fig.add_subplot(gs[:,1:], projection='3d')
axis = format_axes(ax2, "3D projection", "x", "y", "z", [-250, -200, -150, -100, -50, 0], [-250, -200, -150, -100, -50, 0], [-250, -200, -150, -100, -50, 0])
q2 = ax2.quiver(i[0::index], j[0::index], k[0::index], u[0::index], v[0::index], w[0::index], colors=clr3d, length=10)
plt.colorbar(q2)

#fig.colorbar(q1, ax=[ax1, ax3, ax2], cmap='viridis', location='bottom', shrink=0.8, aspect=50).set_ticklabels(([0,0.2,0.4,0.6,0.8,1]), fontsize=16)
plt.savefig(str(savepath+"nonlinear-field-index-"+str(index)+"-niftidata-eg_affine.jpg"), dpi=300)
plt.show()
