# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Common interface for transforms."""
from pathlib import Path
import numpy as np
import h5py
import warnings
from nibabel.loadsave import load as _nbload
from nibabel import funcs as _nbfuncs
from nibabel.nifti1 import intent_codes as INTENT_CODES
from nibabel.cifti2 import Cifti2Image
from scipy import ndimage as ndi

EQUALITY_TOL = 1e-5


class TransformError(TypeError):
    """A custom exception for transforms."""


class SpatialReference:
    """Factory to create spatial references."""

    @staticmethod
    def factory(dataset):
        """Create a reference for spatial transforms."""
        try:
            return SampledSpatialData(dataset)
        except ValueError:
            return ImageGrid(dataset)


class SampledSpatialData:
    """Represent sampled spatial data: regularly gridded (images) and surfaces."""

    __slots__ = ['_ndim', '_coords', '_npoints', '_shape']

    def __init__(self, dataset):
        """Create a sampling reference."""
        self._shape = None

        if isinstance(dataset, SampledSpatialData):
            self._coords = dataset.ndcoords.copy()
            self._npoints, self._ndim = self._coords.shape
            return

        if isinstance(dataset, (str, Path)):
            dataset = _nbload(str(dataset))

        if hasattr(dataset, 'numDA'):  # Looks like a Gifti file
            _das = dataset.get_arrays_from_intent(INTENT_CODES['pointset'])
            if not _das:
                raise TypeError(
                    'Input Gifti file does not contain reference coordinates.')
            self._coords = np.vstack([da.data for da in _das])
            self._npoints, self._ndim = self._coords.shape
            return

        if isinstance(dataset, Cifti2Image):
            raise NotImplementedError

        raise ValueError('Dataset could not be interpreted as an irregular sample.')

    @property
    def npoints(self):
        """Access the total number of voxels."""
        return self._npoints

    @property
    def ndim(self):
        """Access the number of dimensions."""
        return self._ndim

    @property
    def ndcoords(self):
        """List the physical coordinates of this sample."""
        return self._coords

    @property
    def shape(self):
        """Access the space's size of each dimension."""
        return self._shape


class ImageGrid(SampledSpatialData):
    """Class to represent spaces of gridded data (images)."""

    __slots__ = ['_affine', '_inverse', '_ndindex']

    def __init__(self, image):
        """Create a gridded sampling reference."""
        if isinstance(image, (str, Path)):
            image = _nbfuncs.squeeze_image(_nbload(str(image)))

        self._affine = image.affine
        self._shape = image.shape

        self._ndim = getattr(image, 'ndim', len(image.shape))
        if self._ndim == 4:
            self._shape = image.shape[:3]
            self._ndim = 3

        self._npoints = getattr(image, 'npoints',
                                np.prod(self._shape))
        self._ndindex = None
        self._coords = None
        self._inverse = getattr(image, 'inverse',
                                np.linalg.inv(image.affine))

    @property
    def affine(self):
        """Access the indexes-to-RAS affine."""
        return self._affine

    @property
    def inverse(self):
        """Access the RAS-to-indexes affine."""
        return self._inverse

    @property
    def ndindex(self):
        """List the indexes corresponding to the space grid."""
        if self._ndindex is None:
            indexes = tuple([np.arange(s) for s in self._shape])
            self._ndindex = np.array(np.meshgrid(
                *indexes, indexing='ij')).reshape(self._ndim, self._npoints)
        return self._ndindex

    @property
    def ndcoords(self):
        """List the physical coordinates of this gridded space samples."""
        if self._coords is None:
            self._coords = np.tensordot(
                self._affine,
                np.vstack((self.ndindex, np.ones((1, self._npoints)))),
                axes=1
            )[:3, ...]
        return self._coords

    def ras(self, ijk):
        """Get RAS+ coordinates from input indexes."""
        return _apply_affine(ijk, self._affine, self._ndim)

    def index(self, x):
        """Get the image array's indexes corresponding to coordinates."""
        return _apply_affine(x, self._inverse, self._ndim)

    def _to_hdf5(self, group):
        group.attrs['Type'] = 'image'
        group.attrs['ndim'] = self.ndim
        group.create_dataset('affine', data=self.affine)
        group.create_dataset('shape', data=self.shape)

    def __eq__(self, other):
        """Overload equals operator."""
        return (np.allclose(self.affine, other.affine, rtol=EQUALITY_TOL)
                and self.shape == other.shape)

    def __ne__(self, other):
        """Overload not equal operator."""
        return not self == other


class TransformBase:
    """Abstract image class to represent transforms."""

    __slots__ = ('_reference', )

    def __init__(self, reference=None):
        """Instantiate a transform."""
        self._reference = None
        if reference:
            self.reference = reference

    def __call__(self, x, inverse=False):
        """Apply y = f(x)."""
        return self.map(x, inverse=inverse)

    def __add__(self, b):
        """
        Compose this and other transforms.

        Example
        -------
        >>> T1 = TransformBase()
        >>> added = T1 + TransformBase()
        >>> len(added.transforms)
        2

        """
        from .manip import TransformChain
        return TransformChain(transforms=[self, b])

    @property
    def reference(self):
        """Access a reference space where data will be resampled onto."""
        if self._reference is None:
            warnings.warn('Reference space not set')
        return self._reference

    @reference.setter
    def reference(self, image):
        self._reference = ImageGrid(image)

    @property
    def ndim(self):
        """Access the dimensions of the reference space."""
        return self.reference.ndim

    def apply(self, spatialimage, reference=None,
              order=3, mode='constant', cval=0.0, prefilter=True, output_dtype=None):
        """
        Apply a transformation to an image, resampling on the reference spatial object.

        Parameters
        ----------
        spatialimage : `spatialimage`
            The image object containing the data to be resampled in reference
            space
        reference : spatial object, optional
            The image, surface, or combination thereof containing the coordinates
            of samples that will be sampled.
        order : int, optional
            The order of the spline interpolation, default is 3.
            The order has to be in the range 0-5.
        mode : {'constant', 'reflect', 'nearest', 'mirror', 'wrap'}, optional
            Determines how the input image is extended when the resamplings overflows
            a border. Default is 'constant'.
        cval : float, optional
            Constant value for ``mode='constant'``. Default is 0.0.
        prefilter: bool, optional
            Determines if the image's data array is prefiltered with
            a spline filter before interpolation. The default is ``True``,
            which will create a temporary *float64* array of filtered values
            if *order > 1*. If setting this to ``False``, the output will be
            slightly blurred if *order > 1*, unless the input is prefiltered,
            i.e. it is the result of calling the spline filter on the original
            input.

        Returns
        -------
        resampled : `spatialimage` or ndarray
            The data imaged after resampling to reference space.

        """
        if reference is not None and isinstance(reference, (str, Path)):
            reference = _nbload(str(reference))

        _ref = self.reference if reference is None \
            else SpatialReference.factory(reference)

        if isinstance(spatialimage, (str, Path)):
            spatialimage = _nbload(str(spatialimage))

        data = np.asanyarray(spatialimage.dataobj)
        output_dtype = output_dtype or data.dtype
        targets = ImageGrid(spatialimage).index(  # data should be an image
            _as_homogeneous(self.map(_ref.ndcoords.T),
                            dim=_ref.ndim))

        resampled = ndi.map_coordinates(
            data,
            targets.T,
            output=output_dtype,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
        )

        if isinstance(_ref, ImageGrid):  # If reference is grid, reshape
            moved = spatialimage.__class__(
                resampled.reshape(_ref.shape),
                _ref.affine, spatialimage.header)
            moved.header.set_data_dtype(output_dtype)
            return moved

        return resampled

    def map(self, x, inverse=False):
        r"""
        Apply :math:`y = f(x)`.

        TransformBase implements the identity transform.

        Parameters
        ----------
        x : N x D numpy.ndarray
            Input RAS+ coordinates (i.e., physical coordinates).
        inverse : bool
            If ``True``, apply the inverse transform :math:`x = f^{-1}(y)`.

        Returns
        -------
        y : N x D numpy.ndarray
            Transformed (mapped) RAS+ coordinates (i.e., physical coordinates).

        """
        return x

    def to_filename(self, filename, fmt='X5'):
        """Store the transform in BIDS-Transforms HDF5 file format (.x5)."""
        with h5py.File(filename, 'w') as out_file:
            out_file.attrs['Format'] = 'X5'
            out_file.attrs['Version'] = np.uint16(1)
            root = out_file.create_group('/0')
            self._to_hdf5(root)

        return filename

    def _to_hdf5(self, x5_root):
        """Serialize this object into the x5 file format."""
        raise NotImplementedError


def _as_homogeneous(xyz, dtype='float32', dim=3):
    """
    Convert 2D and 3D coordinates into homogeneous coordinates.

    Examples
    --------
    >>> _as_homogeneous((4, 5), dtype='int8', dim=2).tolist()
    [[4, 5, 1]]

    >>> _as_homogeneous((4, 5, 6),dtype='int8').tolist()
    [[4, 5, 6, 1]]

    >>> _as_homogeneous((4, 5, 6, 1),dtype='int8').tolist()
    [[4, 5, 6, 1]]

    >>> _as_homogeneous([(1, 2, 3), (4, 5, 6)]).tolist()
    [[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]]


    """
    xyz = np.atleast_2d(np.array(xyz, dtype=dtype))
    if np.shape(xyz)[-1] == dim + 1:
        return xyz

    return np.hstack((xyz, np.ones((xyz.shape[0], 1), dtype=dtype)))


def _apply_affine(x, affine, dim):
    """Get the image array's indexes corresponding to coordinates."""
    return affine.dot(_as_homogeneous(x, dim=dim).T)[:dim, ...].T
