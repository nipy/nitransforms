import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pathlib import Path
from itertools import zip_longest

from nitransforms.base import TransformError
from nitransforms.linear import Affine
from nitransforms.nonlinear import DenseFieldTransform

class PlotDenseField():
    """
    NotImplented: description of class object here
    """

    __slots__ = ('_path_to_file', '_xfm')

    def __init__(self, path_to_file, is_deltas=True):
        self._path_to_file = path_to_file
        self._xfm = DenseFieldTransform(
            self._path_to_file,
            is_deltas=is_deltas,
        )

        if self._xfm._field.shape[-1] != self._xfm.ndim:
            raise TransformError(
                "The number of components of the field (%d) does not match "
                "the number of dimensions (%d)" % (self._xfm._field.shape[-1], self._xfm.ndim)
            )

    def plot_field(self, scaling=1, index=100, save_to_path=None):
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
        >>> plot = PlotDenseField(
        ...     test_dir / "someones_displacement_field.nii.gz"
        ... ).plot_densefield()
        """

        """Plot"""
        axes = format_fig(
            figsize=(15,8), #(20, 5) if include 3d plot
            gs_rows=2,
            gs_cols=3, #change to 5 if include 3d plot, un-hash in format_axes
            suptitle="Non-Linear DenseFieldTransform field"
            )

        titles=["RGB", None, "Distortion Grid", None, "Quiver", None]
        for i in range(len(axes)):
            ylabel = "y" if i%2==0 else "z"
            format_axes(axes[i], title=titles[i], xlabel="x", ylabel=ylabel)
        
        self.plot_grid((axes[2], axes[3]), index=index)
        self.plot_quiver([axes[4], axes[5]], index=index, scaling=scaling)
        
        """
        format_axes(axes[6], title="3D projection")
        q6 = axes[6].quiver(x, y, z, u, v, w, colors=clr3d, length=2/scaling)
        #plt.colorbar(q6)
        """

        if save_to_path is not None:
            assert os.path.isdir(os.path.dirname(save_to_path))
            plt.savefig(str(save_to_path), dpi=300)
        else:
            pass
        
    def plot_grid(self, ax, index):
        #return NotImplementedError
        x, y, z, u, v, w = self.map_coords(index*10)

        ax[0].scatter(x+u, y+v, c='k', s=0.25)
        ax[1].scatter(x+u, z+w, c='k', s=0.25)

        """e_x  = np.arange(x.min(), x.max()).astype(int)
        e_y = np.arange(y.min(), y.max()).astype(int)

        warped_x=np.zeros(uv.shape)
        warped_y=np.zeros(uv.shape)

        for i, j in enumerate(e_x):
            warped_x[i] = j + u[i]
        for i, j in enumerate(e_y): 
            warped_y[i] = j + v[i]

        import pdb; pdb.set_trace()
        ax[0].plot([e_x[1:-1]] * len(e_y), warped_y, c='k', lw=0.1)
        ax[0].plot(warped_x, [e_y[1:-1]] * len(e_x), c='k', lw=0.1)
        """
    
    def plot_quiver(self, ax, index, scaling=1): 
        x, y, z, u, v, w = self.map_coords(index)

        magnitude = np.sqrt(u**2 + v**2 + w**2)
        clr_xy = np.hypot(u, v)
        clr_xz = np.hypot(u, w)
        clr3d = plt.cm.viridis(magnitude/magnitude.max())

        try:
            if ax.name=='3d':
                xyz = ax.quiver(x, y, z, u, v, w, colors=clr3d, length=1/scaling)
                plt.colorbar(xyz)
        except:
            xy = ax[0].quiver(x, y, u, v, clr_xy, cmap='viridis', angles='xy', scale_units='xy', scale=scaling)
            xz = ax[1].quiver(x, z, u, w, clr_xz, cmap='viridis', angles='xy', scale_units='xy', scale=scaling)
            plt.colorbar(xy)
            plt.colorbar(xz)

    def plot_scatter(self, ax, index, markersize=25):
        x, y, z, u, v, w = self.map_coords(index)

        warped_x = u + x
        warped_y = v + y
        warped_z = w + z

        ax[0].scatter(warped_x, warped_y, s=markersize)
        clr_xy = np.hypot(u, v)
        clr_xz = np.hypot(u, w)

        xy = ax[0].scatter(warped_x, warped_y, c=clr_xy, s=25)
        xz = ax[1].scatter(warped_x, warped_y, c=clr_xz, s=1)
        plt.colorbar(xy)
        plt.colorbar(xz)

    def map_coords(self, index):
        """Calculate vector components of the field using the reference coordinates"""
        x = self._xfm.reference.ndcoords[0][0::index]
        y = self._xfm.reference.ndcoords[1][0::index]
        z = self._xfm.reference.ndcoords[2][0::index]

        u = self._xfm._field[...,0].flatten()[0::index] - x
        v = self._xfm._field[...,1].flatten()[0::index] - y
        w = self._xfm._field[...,2].flatten()[0::index] - z
        return x, y, z, u, v, w


"""Formatting"""

def format_fig(figsize, gs_rows, gs_cols, **kwargs):
    """
    NotImplemented: some description, list of kwargs needed
    """
    params={'gs_wspace':1/3,
            'gs_hspace':1/3,
            'suptitle':None,
            }
    params.update(kwargs)

    fig = plt.figure(figsize=figsize) #(12, 6) for gs(2,3)
    fig.suptitle(str("Non-Linear DenseFieldTransform field"), fontsize='20', weight='bold')
    gs = GridSpec(gs_rows, gs_cols, figure=fig, wspace=params['gs_wspace'], hspace=params['gs_hspace'])

    axes=[]
    for j in range(0, 3):
        for i in range(0, gs_rows):
            axes.append(fig.add_subplot(gs[i,j]))
    #axes.append(fig.add_subplot(gs[:,3:], projection='3d'))
    return axes

def format_axes(axis, **kwargs):
    params={
        'title':None,
        'xlabel':"x",
        'ylabel':"y",
        'zlabel':"z",
        'rotate_3dlabel':False,
        'labelsize':16,
        'ticksize':14,
        }
    params.update(kwargs)

    '''Format the figure axes. For 2D plots, zlabel and zticks parameters are None.'''
    axis.tick_params(labelsize=params['ticksize'])

    axis.set_title(params['title'], weight='bold')
    axis.set_xlabel(params['xlabel'], fontsize=params['labelsize'])
    axis.set_ylabel(params['ylabel'], fontsize=params['labelsize'])

    '''if 3d projection plot'''
    try:
        axis.set_zlabel(params['zlabel'])
        axis.xaxis.set_rotate_label(params['rotate_3dlabel'])
        axis.yaxis.set_rotate_label(params['rotate_3dlabel'])
        axis.zaxis.set_rotate_label(params['rotate_3dlabel'])
    except:
        pass
    return

path_to_file = Path("../tests/data/ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz")
save_to_dir = Path("/Users/julienmarabotto/workspace/Neuroimaging/plots/quiver")

"""___EXAMPLES___"""
"""
#Example 1: plot_template
PlotDenseField(path_to_file, is_deltas=True).plot_field(index=10, save_to_path=str(save_to_dir / "template.jpg"))
plt.show()
"""
"""
#Example 2a: plot_quiver (2d)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
PlotDenseField(path_to_file, is_deltas=True).plot_quiver([axes[0], axes[1]], index=10)
format_axes(axes[0], xlabel="x", ylabel="y", labelsize=14)
format_axes(axes[1], xlabel="x", ylabel="z", labelsize=14)
plt.show()
"""

#Example 2b: plot_quiver (3d)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
PlotDenseField(path_to_file, is_deltas=True).plot_quiver(ax, index=100)
format_axes(ax)
plt.show()

"""
fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
PlotDenseField(path_to_file, is_deltas=True).plot_grid([axes[0], axes[1]], index=100)
plt.show()
"""