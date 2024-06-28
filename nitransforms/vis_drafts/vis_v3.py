import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, Slider

from pathlib import Path
from itertools import product

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

    def show_transform(self, xslice, yslice, zslice, gridstep=10, scaling=1, save_to_path=None):
        """
        Plot output field from DenseFieldTransform class.

        Parameters
        ----------
        xslice: :obj:`int`
            x plane to select for axial projection of the transform.
        yslice: :obj:`int`
            y plane to select for coronary prjection of the transform.
        zslice: :obj:`int`
            z plane to select for sagittal prjection of the transform.
        gridstep: :obj:`int`
            Interval to be used between distortion grid lines (efault: 10). 
        scaling: :obj:`float`
            Fraction by which the quiver plot arrows are to be scaled (default: 1).
        save_to_path: :obj:`str`
            Path to which the output plot is to be saved (default: None).

        Examples
        --------
        >>> PlotDenseField(
        ...     test_dir / "someones_displacement_field.nii.gz"
        ... ).show_transform(50, 50, 50)
        >>> plt.show()

        >>> PlotDenseField(
        ...     path_to_file = test_dir / "someones_displacement_field.nii.gz",
        ...     is_deltas = True,
        ... ).show_transform(
        ...     xslice = 70,
        ...     yslice = 60
        ...     zslice = 90,
        ...     gridstep = 2,
        ...     scaling = 3,
        ...     save_to_path = str(save_to_dir / "template.jpg"),
        ... )
        >>> plt.show()
        """

        fig, axes = format_fig(
            figsize=(10,8),
            gs_rows=3,
            gs_cols=4,
            suptitle="Dense Field Transform \n" + os.path.basename(self._path_to_file),
        )

        projections=["Axial\n(z = "+str(zslice)+")", "Coronary\n(y = "+str(yslice)+")", "Sagittal\n(x = "+str(xslice)+")"]
        for i, ax in enumerate(axes):
            if i < 3:
                xlabel = None
                ylabel = projections[i]
            else:
                xlabel = ylabel = None
            format_axes(ax, xlabel=xlabel, ylabel=ylabel, labelsize=14)
            
        self.plot_grid((axes[2], axes[1], axes[0]), xslice, yslice, zslice, step=gridstep)
        self.plot_deltas((axes[5], axes[4], axes[3]), xslice, yslice, zslice)
        self.plot_quiverdsm((axes[11], axes[10], axes[9], axes[8], axes[7], axes[6]), xslice, yslice, zslice, scaling=scaling)
        
        if save_to_path is not None:
            assert os.path.isdir(os.path.dirname(save_to_path))
            plt.savefig(str(save_to_path), dpi=300)
        else:
            pass
    
    
    def plot_deltas(self, axes, xslice, yslice, zslice):
        planes = self.map_coords(xslice, yslice, zslice)

        for i, j in enumerate(planes):
            x, y, z, u, v, w = j
            c = u + v + w
            
            if i == 0:
                dim1, dim2, vec1, vec2 = y, z, v, w
            elif i == 1:
                dim1, dim2, vec1, vec2 = x, z, u, w
            else:
                dim1, dim2, vec1, vec2 = x, y, u, v

            axes[i].hist2d(dim1, dim2, bins=(50, 50), weights=c, norm=mpl.colors.CenteredNorm(), cmap='seismic')
        

    def plot_quiverdsm(self, ax, xslice, yslice, zslice, scaling=1, three_D=False):
        """
        Plot the Diffusion Scalar Map (dsm) as a quiver plot.
        
        Parameters
        ----------
        axis :obj:`tuple`
            Tuple of three axes on which the dsm should be plotted. Requires THREE axes to illustrate all projections (eg ax1: Axial, ax2: Coronary, ax3: Sagittal)
        xslice: :obj:`int`
            x plane to select for axial projection of the transform.
        yslice: :obj:`int`
            y plane to select for coronary prjection of the transform.
        zslice: :obj:`int`
            z plane to select for sagittal prjection of the transform.
        """
        
        planes = self.map_coords(xslice, yslice, zslice)
        
        if three_D is not False:
            for i, j in enumerate(planes):
                x, y, z, u, v, w = j

            magnitude = np.sqrt(u**2 + v**2 + w**2)
            clr3d = plt.cm.viridis(magnitude/magnitude.max())
            xyz = ax.quiver(x, y, z, u, v, w, colors=clr3d, length=1/scaling)
            plt.colorbar(xyz)
        else:
            for index, plane in enumerate(planes):
                x, y, z, u, v, w = plane
                c_reds, c_greens, c_blues, zeros = [], [], [], []

                ##Optimise here, matrix operations
                for ind, (i, j, k, l, m, n) in enumerate(zip(x, y, z, u, v, w)):
                    if np.abs(u[ind]) > [np.abs(v[ind]) and np.abs(w[ind])]:
                        c_reds.append((i, j, k, l, m, n, u[ind]))
                    elif np.abs(v[ind]) > [np.abs(u[ind]) and np.abs(w[ind])]:
                        c_greens.append((i, j, k, l, m, n, v[ind]))
                    elif np.abs(w[ind]) > [np.abs(u[ind]) and np.abs(v[ind])]:
                        c_blues.append((i, j, k, l, m, n, w[ind]))
                    else:
                        zeros.append(0)

                assert len(np.concatenate((c_reds, c_greens, c_blues))) == len(x) - len(zeros)

                c_reds = np.asanyarray(c_reds)
                c_greens = np.asanyarray(c_greens)
                c_blues = np.asanyarray(c_blues)

                if index == 0:
                    dim1, dim2, vec1, vec2 = 1, 2, 4, 5
                elif index == 1:
                    dim1, dim2, vec1, vec2 = 0, 2, 3, 5
                elif index == 2:
                    dim1, dim2, vec1, vec2 = 0, 1, 3, 4

                ax[index].quiver(c_reds[:, dim1], c_reds[:, dim2], c_reds[:, vec1], c_reds[:, vec2], c_reds[:, -1], cmap='Reds')
                ax[index].quiver(c_greens[:, dim1], c_greens[:, dim2], c_greens[:, vec1], c_greens[:, vec2], c_greens[:, -1], cmap='Greens')
                ax[index].quiver(c_blues[:, dim1], c_blues[:, dim2], c_blues[:, vec1], c_blues[:, vec2], c_blues[:, -1], cmap='Blues')
                ax[index+3].scatter(c_reds[:, dim1], c_reds[:, dim2], c=(c_reds[:, -1]), cmap='bwr', norm=mpl.colors.Normalize(vmin=c_reds.min(), vmax=c_reds.max()), s=1, alpha=1)
                ax[index+3].scatter(c_greens[:, dim1], c_greens[:, dim2], c=(c_greens[:, -1]), cmap='brg', norm=mpl.colors.Normalize(vmin=c_greens.min(), vmax=c_greens.max()), s=1, alpha=1)
                ax[index+3].scatter(c_blues[:, dim1], c_blues[:, dim2], c=(c_blues[:, -1]), cmap='brg_r', norm=mpl.colors.Normalize(vmin=c_blues.min(), vmax=c_blues.max()), s=1, alpha=1)
                
            
    def plot_grid(self, ax, xslice, yslice, zslice, step=10):
        """
        Plot the distortion grid. 

        Parameters
        ----------
        axis :obj:`tuple`
            Tuple of three axes on which the grid should be plotted. Requires THREE axes to illustrate all projections (eg ax1: Axial, ax2: Coronary, ax3: Sagittal)
        xslice: :obj:`int`
            x plane to select for axial projection of the transform.
        yslice: :obj:`int`
            y plane to select for coronary prjection of the transform.
        zslice: :obj:`int`
            z plane to select for sagittal prjection of the transform.
        step: :obj:`int`
            Interval to be used between distortion grid lines (efault: 10). 
        """

        planes = self.map_coords(xslice, yslice, zslice)
        
        for index, plane in enumerate(planes):
            x,y,z,u,v,w = plane

            if index == 0:
                dim1, dim2, vec1, vec2 = y, z, v, w
            elif index == 1:
                dim1, dim2, vec1, vec2 = x, z, u, w
            else:
                dim1, dim2, vec1, vec2 = x, y, u, v

            gc_xy, lenx, leny = get_2dcenters(dim1, dim2, step=step)
            xy = list(gc_xy)

            axx = vec1[0::int(len(vec1)/(lenx * leny))]
            axy = vec2[0::int(len(vec2)/(lenx * leny))]
            
            x_moved, y_moved = [], []
            for ind, (i, j) in enumerate(zip(axx, axy)):
                try:
                    x_moved.append(xy[0][ind] + i)
                    y_moved.append(xy[1][ind] + j)
                except IndexError:
                    break

            for ind, i in enumerate(x_moved):
                if ind%leny == 0:
                    ax[index].plot(x_moved[ind:leny+ind], y_moved[ind:leny+ind], c='k', lw=0.1)
                ax[index].plot(x_moved[ind::leny], y_moved[ind::leny], c='k', lw=0.1)

    def map_coords(self, xslice, yslice, zslice):
        planes = [0]*3
        slices = [
            [False, False, False, False],  # [:,:,index]
            [False, False, False, False],  # [:,index,:]
            [False, False, False, False],   # [index,:,:]
        ]
        
        #iterating through the three chosen planes to calculate corresponding coordinates
        for ind, s in enumerate(slices):
            """Calculate vector components of the field using the reference coordinates"""
            
            #indexing for plane selection [x: sagittal, y: coronal, z: axial, dimension]
            s = [xslice, slice(None), slice(None), None] if ind == 0 else s
            s = [slice(None), yslice, slice(None), None] if ind == 1 else s
            s = [slice(None), slice(None), zslice, None] if ind == 2 else s
            #Full 3d quiver:
            s = [slice(None), slice(None), slice(None), None] if xslice == yslice == zslice == None else s

            #computing coordinates within each plane
            x = self._xfm.reference.ndcoords[0].reshape(np.shape(self._xfm._field[...,-1]))[s[0], s[1], s[2]]
            y = self._xfm.reference.ndcoords[1].reshape(np.shape(self._xfm._field[...,-1]))[s[0], s[1], s[2]]
            z = self._xfm.reference.ndcoords[2].reshape(np.shape(self._xfm._field[...,-1]))[s[0], s[1], s[2]]
            u = self._xfm._field[s[0], s[1], s[2], 0] - x
            v = self._xfm._field[s[0], s[1], s[2], 1] - y
            w = self._xfm._field[s[0], s[1], s[2], 2] - z

            x = x.flatten()
            y = y.flatten()
            z = z.flatten()
            u = u.flatten()
            v = v.flatten()
            w = w.flatten()

            #check indexing has retrived correct dimensions
            if ind==0 and xslice!=None:
                assert x.shape == u.shape == np.shape(self._xfm._field[-1,...,-1].flatten())
            elif ind==1 and yslice!=None:
                assert y.shape == v.shape == np.shape(self._xfm._field[:,-1,:,-1].flatten())
            elif ind==2 and zslice!=None:
                assert z.shape == w.shape == np.shape(self._xfm._field[...,-1,-1].flatten())

            #store 3 slices of datapoints, with overall shape [3 x [6 x [data]]]
            planes[ind] = [x, y, z, u, v, w]

        return planes


"""Formatting"""

def get_2dcenters(x, y, step=10):
        samples_x = np.arange(x.min(), x.max(), step=step).astype(int)
        samples_y = np.arange(y.min(), y.max(), step=step).astype(int)

        lenx = len(samples_x)
        leny = len(samples_y)
        return zip(*product(samples_x, samples_y)), lenx, leny

def format_fig(figsize, gs_rows, gs_cols, **kwargs):
    params={'gs_wspace':0,
            'gs_hspace':1/8,
            'suptitle':None,
            }
    params.update(kwargs)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        params['suptitle'],
        fontsize='20',
        weight='bold')
    
    gs = GridSpec(
        gs_rows,
        gs_cols,
        figure=fig,
        wspace=params['gs_wspace'],
        hspace=params['gs_hspace']
    )

    axes=[]
    for j in range(0, gs_cols):
        for i in range(0, gs_rows):
            axes.append(fig.add_subplot(gs[i,j]))
    return fig, axes

def format_axes(axis, **kwargs):
    params={
        'title':None,
        'xlabel':"x",
        'ylabel':"y",
        'zlabel':"z",
        'xticks':[],
        'yticks':[],
        'zticks':[],
        'rotate_3dlabel':False,
        'labelsize':16,
        'ticksize':14,
        }
    params.update(kwargs)

    '''Format the figure axes. For 2D plots, zlabel and zticks parameters are None.'''
    #axis.tick_params(labelsize=params['ticksize'])
    axis.set_title(params['title'], weight='bold')
    axis.set_xticks(params['xticks'])
    axis.set_yticks(params['yticks'])
    axis.set_xlabel(params['xlabel'], fontsize=params['labelsize'])
    axis.set_ylabel(params['ylabel'], fontsize=params['labelsize'])

    '''if 3d projection plot'''
    try:
        axis.set_zticks(params['zticks'])
        axis.set_zlabel(params['zlabel'])
        axis.xaxis.set_rotate_label(params['rotate_3dlabel'])
        axis.yaxis.set_rotate_label(params['rotate_3dlabel'])
        axis.zaxis.set_rotate_label(params['rotate_3dlabel'])
    except:
        pass
    return

path_to_file = Path("/Users/julienmarabotto/workspace/nitransforms/nitransforms/tests/data/ds-005_sub-01_from-OASIS_to-T1_warp_fsl.nii.gz")
save_to_dir = Path("/Users/julienmarabotto/workspace/Neuroimaging/plots/quiver")

"""___EXAMPLES___"""

#Example 1: plot_template
PlotDenseField(path_to_file, is_deltas=True).show_transform(
    xslice=50,
    yslice=75,
    zslice=90,
    gridstep=5,
    save_to_path=str(save_to_dir / "template_v3.jpg")
    #save_to_path=None
)
plt.show()

"""
#Example 2a: plot_quiver (2d)
fig, axes = plt.subplots(1, 3, figsize=(10, 4), tight_layout=True)
PlotDenseField(path_to_file, is_deltas=True).plot_grid(
    [axes[2], axes[1], axes[0]],
    xslice=50,
    yslice=75,
    zslice=90,
)
plt.show()
"""
"""
#Example 2b: plot_quiver (3d)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
PlotDenseField(path_to_file, is_deltas=True).plot_quiverdsm(
    ax,
    xslice=None,
    yslice=None,
    zslice=None,
    scaling=10,
    three_D=True,
)
format_axes(ax) #, xticks=[-250, -200, -150, -100, -50, 0], yticks=[-200, -150, -100, -50, 0], zticks=[-200, -150, -100, -50, 0]
plt.show()
"""
