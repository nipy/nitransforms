# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Linear transforms."""
import warnings
import numpy as np

from .base import ImageGrid, TransformBase, _as_homogeneous, EQUALITY_TOL
from . import io


class Affine(TransformBase):
    """Represents linear transforms on image data."""

    __slots__ = ['_matrix']

    def __init__(self, matrix=None, reference=None):
        """
        Initialize a linear transform.

        Parameters
        ----------
        matrix : ndarray
            The inverse coordinate transformation matrix **in physical
            coordinates**, mapping coordinates from *reference* space
            into *moving* space.
            This matrix should be provided in homogeneous coordinates.

        Examples
        --------
        >>> xfm = Affine(reference=datadir / 'someones_anatomy.nii.gz')
        >>> xfm.matrix  # doctest: +NORMALIZE_WHITESPACE
        array([[[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]]])

        >>> xfm = Affine([[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        >>> xfm.matrix  # doctest: +NORMALIZE_WHITESPACE
        array([[[1, 0, 0, 4],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]])

        """
        super(Affine, self).__init__()
        if matrix is None:
            matrix = [np.eye(4)]

        if np.ndim(matrix) == 2:
            matrix = [matrix]

        self._matrix = np.array(matrix)
        assert self._matrix.ndim == 3, 'affine matrix should be 3D'
        assert (
            self._matrix.shape[-2] == self._matrix.shape[-1]
        ), 'affine matrix is not square'

        if reference:
            self.reference = reference

    def __eq__(self, other):
        """
        Overload equals operator.

        Examples
        --------
        >>> xfm1 = Affine([[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        >>> xfm2 = Affine(xfm1.matrix)
        >>> xfm1 == xfm2
        True

        """
        _eq = np.allclose(self.matrix, other.matrix, rtol=EQUALITY_TOL)
        if _eq and self._reference != other._reference:
            warnings.warn('Affines are equal, but references do not match.')
        return _eq

    @property
    def matrix(self):
        """Access the internal representation of this affine."""
        return self._matrix

    def map(self, x, inverse=False, index=0):
        r"""
        Apply :math:`y = f(x)`.

        Parameters
        ----------
        x : N x D numpy.ndarray
            Input RAS+ coordinates (i.e., physical coordinates).
        inverse : bool
            If ``True``, apply the inverse transform :math:`x = f^{-1}(y)`.
        index : int, optional
            Transformation index

        Returns
        -------
        y : N x D numpy.ndarray
            Transformed (mapped) RAS+ coordinates (i.e., physical coordinates).

        Examples
        --------
        >>> xfm = Affine([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
        >>> xfm.map((0,0,0))
        array([[1., 2., 3.]])

        >>> xfm.map((0,0,0), inverse=True)
        array([[-1., -2., -3.]])

        """
        coords = _as_homogeneous(x, dim=self._matrix[0].shape[0] - 1).T
        affine = self._matrix[index]
        if inverse is True:
            affine = np.linalg.inv(self._matrix[index])
        return affine.dot(coords).T[..., :-1]

    def _to_hdf5(self, x5_root):
        """Serialize this object into the x5 file format."""
        xform = x5_root.create_dataset('Transform', data=self._matrix)
        xform.attrs['Type'] = 'affine'
        x5_root.create_dataset('Inverse', data=np.linalg.inv(self._matrix))

        if self._reference:
            self.reference._to_hdf5(x5_root.create_group('Reference'))

    def to_filename(self, filename, fmt='X5', moving=None):
        """Store the transform in BIDS-Transforms HDF5 file format (.x5)."""
        if fmt.lower() in ['itk', 'ants', 'elastix']:
            itkobj = io.itk.ITKLinearTransformArray.from_ras(self.matrix)
            itkobj.to_filename(filename)
            return filename

        # Rest of the formats peek into moving and reference image grids
        if moving is not None:
            moving = ImageGrid(moving)
        else:
            moving = self.reference

        if fmt.lower() == 'afni':
            afniobj = io.afni.AFNILinearTransformArray.from_ras(
                self.matrix, moving=moving, reference=self.reference)
            afniobj.to_filename(filename)
            return filename

        if fmt.lower() == 'fsl':
            fslobj = io.fsl.FSLLinearTransformArray.from_ras(
                self.matrix, moving=moving, reference=self.reference
            )
            fslobj.to_filename(filename)
            return filename

        if fmt.lower() == 'fs':
            # xform info
            lt = io.LinearTransform()
            lt['sigma'] = 1.
            lt['m_L'] = self.matrix
            # Just for reference, nitransforms does not write VOX2VOX
            lt['src'] = io.VolumeGeometry.from_image(moving)
            lt['dst'] = io.VolumeGeometry.from_image(self.reference)
            # to make LTA file format
            lta = io.LinearTransformArray()
            lta['type'] = 1  # RAS2RAS
            lta['xforms'].append(lt)

            with open(filename, 'w') as f:
                f.write(lta.to_string())
            return filename

        raise NotImplementedError

    @classmethod
    def from_filename(cls, filename, fmt='X5',
                      reference=None, moving=None):
        """Create an affine from a transform file."""
        if fmt.lower() in ('itk', 'ants', 'elastix'):
            _factory = io.itk.ITKLinearTransformArray
        elif fmt.lower() in ('lta', 'fs'):
            _factory = io.LinearTransformArray
        else:
            raise NotImplementedError

        struct = _factory.from_filename(filename)
        matrix = struct.to_ras(reference=reference, moving=moving)

        return cls(matrix, reference=reference)


load = Affine.from_filename
