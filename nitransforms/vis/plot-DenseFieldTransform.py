import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pathlib import Path

from nitransforms.base import TransformError
from nitransforms.linear import Affine
from nitransforms.nonlinear import DenseFieldTransform

class Vis():

    __slots__ = ('_path_to_file')

    def __init__(self, path_to_file):
         self._path_to_file = path_to_file

    def plot_densefield(self, is_deltas=True, scaling=1, index=1000, save_to_dir=None):
        """
        Plot output field from DenseFieldTransform class. 

        Parameters
        ----------
        is_deltas : :obj:`bool`
            Whether this is a displacements (deltas) field (default: is_deltas=True), or deformations (is_deltas=False).
        save_to_path: :obj:`str`
            Path to which the output plot is to be saved.
        scaling: :obj:`float`
            Fraction by which the quiver plot arrows are to be scaled (default: 1)
        index: :obj:`float`
            Indexing for plotting (default: index=100). The index defines the interval to be used when selecting datapoints, such that are only plotted elements [0::index]

        Example
        -------
        >>> plot = Vis(
        ...     test_dir / "someones_displacement_field.nii.gz"
        ... ).plot_densefield()        
        

        >>> plot = Vis(
        ...     test_dir / "someones_displacement_field.nii.gz"
        ... ).plot_densefield(
        ...     is_deltas = True #deltas field
                scaling = 0.25 #arrow scaling = 4 times true length
                index = 10 #plot 1/10 data points, with indexing [0::10]
        ...     save_to_path = test_dir / "plot_of_someones_displacement_field.nii.gz" #save figure
        ... )
        """
        
        xfm = DenseFieldTransform(
             self._path_to_file,
             is_deltas=is_deltas,
        )

        if xfm._field.shape[-1] != xfm.ndim:
            raise TransformError(
                "The number of components of the field (%d) does not match "
                "the number of dimensions (%d)" % (xfm._field.shape[-1], xfm.ndim)
            )

        x, y, z, u, v, w = self.map_coords(xfm)
        magnitude = np.sqrt(u**2 + v**2 + w**2)
        clr_xy = np.hypot(u, v)[0::index]
        clr_xz = np.hypot(u, w)[0::index]
        clr3d = plt.cm.viridis(magnitude[0::index]/magnitude[0::index].max())

        """Plot"""
        fig, gs = self.format_fig(figsize=(15, 8), gs_rows=2, gs_cols=3, gs_wspace=1/4, gs_hspace=1/2.5)

        ax1 = fig.add_subplot(gs[0,0])
        ax_params = self.format_axes(ax1, "x-y projection", "x", "y")
        q1 = ax1.quiver(x[0::index], y[0::index], u[0::index], v[0::index], clr_xy, cmap='viridis', angles='xy', scale_units='xy', scale=scaling)
        plt.colorbar(q1)

        ax2 = fig.add_subplot(gs[1,0])
        ax_params = self.format_axes(ax2, "x-z projection", "x", "z")
        q2 = ax2.quiver(x[0::index], z[0::index], u[0::index], w[0::index], clr_xz, cmap='viridis', angles='xy', scale_units='xy', scale=scaling)
        plt.colorbar(q2)

        ax3 = fig.add_subplot(gs[:,1:], projection='3d')
        ax_params = self.format_axes(ax3, "3D projection", "x", "y", "z")
        q3 = ax3.quiver(x[0::index], y[0::index], z[0::index], u[0::index], v[0::index], w[0::index], colors=clr3d, length=2/scaling)
        plt.colorbar(q3)

        if save_to_dir is not None:
            plt.savefig(str(save_to_dir), dpi=300)
            assert os.path.isdir(os.path.dirname(save_to_dir))
        else:
            pass
        plt.show()

    def map_coords(self, xfm): 
        """Calculate vector components of the field using the reference coordinates"""
        x = xfm.reference.ndcoords[0]
        y = xfm.reference.ndcoords[1]
        z = xfm.reference.ndcoords[2]

        u = xfm._field[...,0].flatten() - x
        v = xfm._field[...,1].flatten() - y
        w = xfm._field[...,2].flatten() - z
        return x, y, z, u, v, w
    
    def format_fig(self, figsize, gs_rows, gs_cols, gs_wspace, gs_hspace):
        fig = plt.figure(figsize=figsize) #(12, 6) for gs(2,3)
        fig.suptitle(str("Non-Linear DenseFieldTransform field"), fontsize='20', weight='bold')
        gs = GridSpec(gs_rows, gs_cols, figure=fig, wspace=gs_wspace, hspace=gs_hspace)
        return fig, gs

    def format_axes(self, axis, title=None, xlabel="x", ylabel="y", zlabel="z", rotate_3dlabel=False, labelsize=16, ticksize=14):
        '''Format the figure axes. For 2D plots, zlabel and zticks parameters are None.'''
        axis.tick_params(labelsize=ticksize)

        axis.set_title(title, weight='bold')
        axis.set_xlabel(xlabel, fontsize=labelsize)
        axis.set_ylabel(ylabel, fontsize=labelsize)

        '''if 3d projection plot'''
        try:
            axis.set_zlabel(zlabel, fontsize=labelsize+4)
            axis.xaxis.set_rotate_label(rotate_3dlabel)
            axis.yaxis.set_rotate_label(rotate_3dlabel)
            axis.zaxis.set_rotate_label(rotate_3dlabel)
        except:
            pass
        return
    
    def format_ticks(self, axis, xticks, yticks, zticks):
        axis.set_xticks((xticks))
        axis.set_yticks((yticks))
        try:
            axis.set_zticks((zticks))
        except:
            pass

    
#Example:
path_to_file = Path("../tests/data/ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz")
save_to_dir = Path("/Users/julienmarabotto/workspace/Neuroimaging/plots/quiver")

plot = Vis(path_to_file).plot_densefield(is_deltas=True, scaling=0.25, save_to_dir=(save_to_dir / "example_dense_field.jpg"), index=10000)
