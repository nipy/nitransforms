import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nb

from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, Slider

from pathlib import Path
from itertools import product

from nitransforms.base import TransformError
from nitransforms.nonlinear import DenseFieldTransform

class PlotDenseField():
    """
    Vizualisation of a transformation file using nitransform's DenseFielTransform module. Generates four sorts of plots:
        i) the deformed grid\n
        ii) the normalised deformation field density map\n
        iii) the quiver map of the field, coloured according to its diffusion scalar map\n
        iv) the quiver map of the field, coloured according to the jacobian of the coordinate matrices\n
    for 3 image projections:
        i) axial (fixed z slice)\n
        ii) saggital (fixed y slice)\n
        iii) coronal (fixed x slice)\n
    Outputs the resulting 3 x 4 image grid.

    Parameters
    ----------

    path_to_file: :obj:`str`
        Path from which the trasnformation file should be read.
    is_deltas: :obj:`bool`
        Whether the field is a displacement field or a deformations field. Default = True
    
    Example:
    path_to_file = Path("/test-directory/someones-anatomy.nii.gz")
    PlotDenseField(path_to_file=path_to_file, is_deltas=True).show_transform(
        xslice=50,
        yslice=75,
        zslice=90,
        gridstep=5,
        save_to_path=str("test-directory/vis-someones-anatomy.jpg")
    )
    plt.show()
    """
    __slots__ = ('_path_to_file', '_xfm', '_voxel_size')

    def __init__(self, path_to_file, is_deltas=True):
        self._path_to_file = path_to_file
        self._xfm = DenseFieldTransform(
            self._path_to_file,
            is_deltas=is_deltas,
        )
        self._voxel_size = nb.load(path_to_file).header.get_zooms()

        if self._xfm._field.shape[-1] != self._xfm.ndim:
            raise TransformError(
                "The number of components of the field (%d) does not match "
                "the number of dimensions (%d)" % (self._xfm._field.shape[-1], self._xfm.ndim)
            )

    def show_transform(self, xslice, yslice, zslice, scaling=1, show_brain=True, show_grid=True, lw=0.1, save_to_path=None):
        """
        Plot output field from DenseFieldTransform class.

        Parameters
        ----------
        xslice: :obj:`int`
            x plane to select for axial projection of the transform.
        yslice: :obj:`int`
            y plane to select for coronal prjection of the transform.
        zslice: :obj:`int`
            z plane to select for sagittal prjection of the transform.
        scaling: :obj:`float`
            Fraction by which the quiver plot arrows are to be scaled (default: 1).
        save_to_path: :obj:`str`
            Path to which the output plot is to be saved (default: None).

        Examples
        --------
        >>> PlotDenseField(
        ...     test_dir / "someones-displacement-field.nii.gz"
        ... ).show_transform(50, 50, 50)
        >>> plt.show()

        >>> PlotDenseField(
        ...     path_to_file = "test_dir/someones-displacement-field.nii.gz",
        ...     is_deltas = True,
        ... ).show_transform(
        ...     xslice = 70,
        ...     yslice = 60
        ...     zslice = 90,
        ...     scaling = 3,
        ...     show_brain=False,
        ...     lw = 0.2
        ...     save_to_path = str("test_dir/my_file.jpg"),
        ... )
        >>> plt.show()
        """
        fig, axes = format_fig(
            figsize=(9,9),
            gs_rows=3,
            gs_cols=3,
            suptitle="Dense Field Transform \n" + os.path.basename(self._path_to_file),
        )
        fig.subplots_adjust(bottom=0.15)

        projections=["Axial", "Coronal", "Sagittal"]
        for i, ax in enumerate(axes):
            if i < 3:
                xlabel = None
                ylabel = projections[i]
            else:
                xlabel = ylabel = None
            format_axes(ax, xlabel=xlabel, ylabel=ylabel, labelsize=16)

        self.plot_distortion((axes[2], axes[1], axes[0]), xslice, yslice, zslice, show_grid=show_grid, show_brain=show_brain, lw=lw, show_titles=False)
        self.plot_quiverdsm((axes[5], axes[4], axes[3]), xslice, yslice, zslice, scaling=scaling, show_titles=False)
        self.plot_jacobian((axes[8], axes[7], axes[6]), xslice, yslice, zslice, show_titles=False)

        sliders = self.sliders(fig, xslice, yslice, zslice)
        #NotImplemented: Interactive slider update here:

        if save_to_path is not None:
            assert os.path.isdir(os.path.dirname(save_to_path))
            plt.savefig(str(save_to_path), dpi=300)
        else:
            pass
    
    
    def plot_distortion(self, axes, xslice, yslice, zslice, show_brain=True, show_grid=True, lw=0.1, show_titles=True):
        """
        Plot the distortion grid. 

        Parameters
        ----------
        axis :obj:`tuple`
            Axes on which the grid should be plotted. Requires 3 axes to illustrate all projections (eg ax1: Axial, ax2: coronal, ax3: Sagittal)
        xslice: :obj:`int`
            x plane to select for axial projection of the transform.
        yslice: :obj:`int`
            y plane to select for coronal prjection of the transform.
        zslice: :obj:`int`
            z plane to select for sagittal prjection of the transform.
        show_brain: :obj:`bool`
            Whether the normalised density map of the distortions should be plotted (Default: True).
        show_grid: :obj:`bool`
            Whether the distorted grid lines should be plotted (Default: True).
        lw: :obj:`float`
            Line width used for gridlines (Default: 0.1).

        Example:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
        PlotDenseField(
            path_to_file="test_dir/someones-displacement-field.nii.gz",
            is_deltas=True,
        ).plot_distortion(
            axes=[axes[2], axes[1], axes[0]],
            xslice=50,
            yslice=75,
            zslice=90,
            show_brain=True,
            show_grid=True,
            lw=0.2,
        )
        plt.savefig(str("test_dir/deformationgrid.jpg", dpi=300)
        plt.show()
        """
        planes, titles = self.get_planes(xslice, yslice, zslice)

        for index, plane in enumerate(planes):
            x,y,z,u,v,w = plane
            shape = self._xfm._field.shape[:-1]

            if index == 0:
                dim1, dim2, vec1, vec2 = y, z, v, w
                len1, len2 = shape[1], shape[2]
            elif index == 1:
                dim1, dim2, vec1, vec2 = x, z, u, w
                len1, len2 = shape[0], shape[2]
            else:
                dim1, dim2, vec1, vec2 = x, y, u, v
                len1, len2 = shape[0], shape[1]

            c = np.sqrt(vec1**2 + vec2**2)
            c = c/c.max()

            if show_grid==True:
                x_moved = dim1+vec1
                y_moved = dim2+vec2

                for idx in range(0, len1, 1):
                    axes[index].plot(x_moved[idx*len2:(idx+1)*len2], y_moved[idx*len2:(idx+1)*len2], c='k', lw=lw)
                for idx in range(0, len2, 1):
                    axes[index].plot(x_moved[idx::len2], y_moved[idx::len2], c='k', lw=lw)

            if show_brain==True:
                axes[index].scatter(dim1, dim2, c=c, cmap='RdPu')
            
            if show_titles==True:
                axes[index].set_title(titles[index], fontsize=14, weight='bold')


    def plot_quiverdsm(self, axes, xslice, yslice, zslice, scaling=1, three_D=False, show_titles=True):
        """
        Plot the Diffusion Scalar Map (dsm) as a quiver plot.

        Parameters
        ----------
        axis :obj:`tuple`
            Axes on which the quiver should be plotted. Requires 3 axes to illustrate the dsm mapped as a quiver plot for each projection.
        xslice: :obj:`int`
            x plane to select for axial projection of the transform.
        yslice: :obj:`int`
            y plane to select for coronal projection of the transform.
        zslice: :obj:`int`
            z plane to select for sagittal projection of the transform.
        scaling: :obj:`float`
            Fraction by which the quiver plot arrows are to be scaled (default: 1).
        three_D: :obj:`bool`
            Whether the quiver plot is to be projected onto a 3D axis (default: False)
        
        Example:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
        PlotDenseField(
            path_to_file="test_dir/someones-displacement-field.nii.gz",
            is_deltas=True,
        ).plot_quiverdsm(
            axes=[axes[2], axes[1], axes[0]],
            xslice=50,
            yslice=75,
            zslice=90,
            scaling=2,
        )
        plt.savefig(str("test_dir/quiverdsm.jpg", dpi=300)
        plt.show()

        #Example 2: 3D quiver
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
        planes, titles = self.get_planes(xslice, yslice, zslice)
        if three_D is not False:
            raise NotImplementedError("3d Quiver plot not finalised.")
        
            #finalise 3d quiver below:
            for i, j in enumerate(planes):
                x, y, z, u, v, w = j

            magnitude = np.sqrt(u**2 + v**2 + w**2)
            clr3d = plt.cm.viridis(magnitude/magnitude.max())
            xyz = axes.quiver(x, y, z, u, v, w, colors=clr3d, length=1/scaling)
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

                axes[index].quiver(c_reds[:, dim1], c_reds[:, dim2], c_reds[:, vec1], c_reds[:, vec2], c_reds[:, -1], cmap='Reds')
                axes[index].quiver(c_greens[:, dim1], c_greens[:, dim2], c_greens[:, vec1], c_greens[:, vec2], c_greens[:, -1], cmap='Greens')
                axes[index].quiver(c_blues[:, dim1], c_blues[:, dim2], c_blues[:, vec1], c_blues[:, vec2], c_blues[:, -1], cmap='Blues')

                if show_titles==True:
                    axes[index].set_title(titles[index], fontsize=14, weight='bold')


    def plot_jacobian(self, axes, xslice, yslice, zslice, show_titles=True):
        """
        Map the divergence of the transformation field using a quiver plot.
        
        Parameters
        ----------
        axis :obj:`tuple`
            Axes on which the quiver should be plotted. Requires 3 axes to illustrate each projection (eg ax1: Axial, ax2: coronal, ax3: Sagittal)
        xslice: :obj:`int`
            x plane to select for axial projection of the transform.
        yslice: :obj:`int`
            y plane to select for coronal projection of the transform.
        zslice: :obj:`int`
            z plane to select for sagittal projection of the transform.

        Example:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
        PlotDenseField(
            path_to_file="test_dir/someones-displacement-field.nii.gz",
            is_deltas=True,
        ).plot_jacobian(
            axes=[axes[2], axes[1], axes[0]],
            xslice=50,
            yslice=75,
            zslice=90,
        )
        plt.savefig(str("test_dir/jacobians.jpg", dpi=300)
        plt.show()
        """
        planes, titles = self.get_planes(xslice, yslice, zslice)
        slices = [
            [False, False, False, False],  # [:,:,index]
            [False, False, False, False],  # [:,index,:]
            [False, False, False, False],  # [index,:,:]
        ]
        jacobians = np.zeros((3), dtype=np.ndarray)

        #iterating through the three chosen planes to calculate corresponding coordinates
        for ind, s in enumerate(slices):
            s = [xslice, slice(None), slice(None), None] if ind == 0 else s
            s = [slice(None), yslice, slice(None), None] if ind == 1 else s
            s = [slice(None), slice(None), zslice, None] if ind == 2 else s
            J = self.get_jacobian().reshape(self._xfm._field[..., -1].shape)[s[0], s[1], s[2]]
            jacobians[ind] = J.flatten()

        for index, plane in enumerate(planes):
            x, y, z, u, v, w = plane

            if index == 0:
                dim1, dim2, vec1, vec2 = y, z, v, w
            elif index == 1:
                dim1, dim2, vec1, vec2 = x, z, u, w
            else:
                dim1, dim2, vec1, vec2 = x, y, u, v

            c = jacobians[index]
            axes[index].scatter(dim1, dim2, c=c, norm=mpl.colors.CenteredNorm(), cmap='seismic')

            if show_titles==True:
                axes[index].set_title(titles[index], fontsize=14, weight='bold')


    def get_coords(self):
            """Calculate vector components of the field using the reference coordinates"""
            x = self._xfm.reference.ndcoords[0].reshape(np.shape(self._xfm._field[...,-1]))
            y = self._xfm.reference.ndcoords[1].reshape(np.shape(self._xfm._field[...,-1]))
            z = self._xfm.reference.ndcoords[2].reshape(np.shape(self._xfm._field[...,-1]))
            u = self._xfm._field[..., 0] - x
            v = self._xfm._field[..., 1] - y
            w = self._xfm._field[..., 2] - z
            return x, y, z, u, v, w


    def get_jacobian(self):
        """Calculate the Jacobian matrix of the field"""
        x, y, z, u, v, w = self.get_coords()
        voxx, voxy, voxz, _ = self._voxel_size

        shape = self._xfm._field[..., -1].shape
        zeros = np.zeros(shape)
        jacobians = zeros.flatten()

        dxdx = (np.diff(u, axis=0) / voxx)
        dydx = (np.diff(v, axis=0) / voxx)
        dzdx = (np.diff(w, axis=0) / voxx)

        dxdy = (np.diff(u, axis=1) / voxy)
        dydy = (np.diff(v, axis=1) / voxy)
        dzdy = (np.diff(w, axis=1) / voxy)

        dxdz = (np.diff(u, axis=2) / voxz)
        dydz = (np.diff(v, axis=2) / voxz)
        dzdz = (np.diff(w, axis=2) / voxz)

        partials = [dxdx, dydx, dzdx, dxdy, dydy, dzdy, dxdz, dydz, dzdz]

        for idx, j in enumerate(partials):
            if idx < 3:
                dim = zeros[-1,:,:][None,:,:]
                ax=0
            elif idx >=3 and idx < 6:
                dim = zeros[:,-1,:][:,None,:]
                ax=1
            elif idx >=6:
                dim = zeros[:,:,-1][:,:,None]
                ax=2

            partials[idx] = np.append(j, dim, axis=ax).flatten()

        dxdx, dydx, dzdx, dxdy, dydy, dzdy, dxdz, dydz, dzdz = partials

        for idx, k in enumerate(jacobians):
            jacobians[idx] = np.linalg.det(
                np.array(
                    [
                        [dxdx[idx], dxdy[idx], dxdz[idx]],
                        [dydx[idx], dydy[idx], dydz[idx]],
                        [dzdx[idx], dzdy[idx], dzdz[idx]]
                    ]
                )
            )
        return jacobians


    def get_planes(self, xslice, yslice, zslice):
        """Define slice selection for visualisation"""
        titles = ["Sagittal", "Coronal", "Axial"]
        planes = [0]*3
        slices = [
            [False, False, False, False],  # [:,:,index]
            [False, False, False, False],  # [:,index,:]
            [False, False, False, False],   # [index,:,:]
        ]

        for ind, s in enumerate(slices):
            x, y, z, u, v, w = self.get_coords()

            #indexing for plane selection [x: sagittal, y: coronal, z: axial, dimension]
            s = [xslice, slice(None), slice(None), None] if ind == 0 else s
            s = [slice(None), yslice, slice(None), None] if ind == 1 else s
            s = [slice(None), slice(None), zslice, None] if ind == 2 else s
            #Full 3d quiver:
            s = [slice(None), slice(None), slice(None), None] if xslice == yslice == zslice == None else s

            #computing coordinates within each plane
            x = x[s[0], s[1], s[2]]
            y = y[s[0], s[1], s[2]]
            z = z[s[0], s[1], s[2]]
            u = self._xfm._field[s[0], s[1], s[2], 0] - x
            v = self._xfm._field[s[0], s[1], s[2], 1] - y
            w = self._xfm._field[s[0], s[1], s[2], 2] - z

            x = x.flatten()
            y = y.flatten()
            z = z.flatten()
            u = u.flatten()
            v = v.flatten()
            w = w.flatten()

            #check indexing has retrieved correct dimensions
            if ind==0 and xslice!=None:
                assert x.shape == u.shape == np.shape(self._xfm._field[-1,...,-1].flatten())
            elif ind==1 and yslice!=None:
                assert y.shape == v.shape == np.shape(self._xfm._field[:,-1,:,-1].flatten())
            elif ind==2 and zslice!=None:
                assert z.shape == w.shape == np.shape(self._xfm._field[...,-1,-1].flatten())

            #store 3 slices of datapoints, with overall shape [3 x [6 x [data]]]
            planes[ind] = [x, y, z, u, v, w]
        return planes, titles


    def sliders(self, fig, xslice, yslice, zslice):
        slices = [
            [zslice, len(self._xfm._field[0][0]), "zslice"],
            [yslice, len(self._xfm._field[0]), "yslice"],
            [xslice, len(self._xfm._field), "xslice"],
            ]
        axes = [
            [1/7, 0.1, 1/7, 0.025],
            [3/7, 0.1, 1/7, 0.025],
            [5/7, 0.1, 1/7, 0.025],
            ]
        sliders = []

        for index, slider_axis in enumerate(axes):
            slice_dim = slices[index][0]
            sax = fig.add_axes(slider_axis)
            slider = Slider(
                ax=sax,
                valmin=0,
                valmax=slices[index][1],
                valinit=slice_dim,
                valstep=1,
                valfmt='%d',
                label=slices[index][2],
                orientation="horizontal"
            )
            sliders.append(slider)

        return sliders


    def update_sliders(self, slider):
        raise NotImplementedError("Sliders not implemented.")
        
        new_slider = slider.val
        return new_slider


def get_2dcenters(x, y, step=2):
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
