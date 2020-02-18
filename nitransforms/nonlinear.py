# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Nonlinear transforms."""
import warnings
import numpy as np
from .base import TransformBase
from . import io

# from .base import ImageGrid
# from nibabel.funcs import four_to_three


class DisplacementsFieldTransform(TransformBase):
    """Represents a dense field of displacements (one vector per voxel)."""

    __slots__ = ['_field']

    def __init__(self, field, reference=None):
        """Create a dense deformation field transform."""
        super().__init__()
        self._field = np.asanyarray(field.dataobj)

        ndim = self._field.ndim - 1
        if self._field.shape[-1] != ndim:
            raise ValueError(
                'The number of components of the displacements (%d) does not '
                'the number of dimensions (%d)' % (self._field.shape[-1], ndim))

        self.reference = field.__class__(np.zeros(self._field.shape[:-1]),
                                         field.affine, field.header)

    def map(self, x, inverse=False):
        r"""
        Apply :math:`y = f(x)`.

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

        Examples
        --------
        >>> field = np.zeros((10, 10, 10, 3))
        >>> field[..., 0] = 4.0
        >>> fieldimg = nb.Nifti1Image(field, np.diag([2., 2., 2., 1.]))
        >>> xfm = DisplacementsFieldTransform(fieldimg)
        >>> xfm([4.0, 4.0, 4.0]).tolist()
        [[8.0, 4.0, 4.0]]

        >>> xfm([[4.0, 4.0, 4.0], [8, 2, 10]]).tolist()
        [[8.0, 4.0, 4.0], [12.0, 2.0, 10.0]]

        """
        if inverse is True:
            raise NotImplementedError
        ijk = self.reference.index(x)
        indexes = np.round(ijk).astype('int')
        if np.any(np.abs(ijk - indexes) > 0.05):
            warnings.warn(
                'Some coordinates are off-grid of the displacements field.')
        indexes = tuple(tuple(i) for i in indexes.T)
        return x + self._field[indexes]

    @classmethod
    def from_filename(cls, filename, fmt='X5'):
        if fmt == 'afni':
            _factory = io.afni.AFNIDisplacementsField
        elif fmt == 'itk':
            _factory = io.itk.ITKDisplacementsField
        else:
            raise NotImplementedError

        return cls(_factory.from_filename(filename))


load = DisplacementsFieldTransform.from_filename

# class BSplineFieldTransform(TransformBase):
#     """Represent a nonlinear transform parameterized by BSpline basis."""

#     __slots__ = ['_coeffs', '_knots', '_refknots', '_order', '_moving']
#     __s = (slice(None), )

#     def __init__(self, reference, coefficients, order=3):
#         """Create a smooth deformation field using B-Spline basis."""
#         super(BSplineFieldTransform, self).__init__()
#         self._order = order
#         self.reference = reference

#         if coefficients.shape[-1] != self.ndim:
#             raise ValueError(
#                 'Number of components of the coefficients does '
#                 'not match the number of dimensions')

#         self._coeffs = np.asanyarray(coefficients.dataobj)
#         self._knots = ImageGrid(four_to_three(coefficients)[0])
#         self._cache_moving()

#     def _cache_moving(self):
#         self._moving = np.zeros((self.reference.shape) + (3, ),
#                                 dtype='float32')
#         ijk = np.moveaxis(self.reference.ndindex, 0, -1).reshape(-1, self.ndim)
#         xyz = np.moveaxis(self.reference.ndcoords, 0, -1).reshape(-1, self.ndim)
#         print(np.shape(xyz))

#         for i in range(np.shape(xyz)[0]):
#             print(i, xyz[i, :])
#             self._moving[tuple(ijk[i]) + self.__s] = self._interp_transform(xyz[i, :])

#     def _interp_transform(self, coords):
#         # Calculate position in the grid of control points
#         knots_ijk = self._knots.inverse.dot(np.hstack((coords, 1)))[:3]
#         neighbors = []
#         offset = 0.0 if self._order & 1 else 0.5
#         # Calculate neighbors along each dimension
#         for dim in range(self.ndim):
#             first = int(np.floor(knots_ijk[dim] + offset) - self._order // 2)
#             neighbors.append(list(range(first, first + self._order + 1)))

#         # Get indexes of the neighborings clique
#         ndindex = np.moveaxis(
#             np.array(np.meshgrid(*neighbors, indexing='ij')), 0, -1).reshape(
#             -1, self.ndim)

#         # Calculate the tensor B-spline weights of each neighbor
#         # weights = np.prod(vbspl(ndindex - knots_ijk), axis=-1)
#         ndindex = [tuple(v) for v in ndindex]

#         # Retrieve coefficients and deal with boundary conditions
#         zero = np.zeros(self.ndim)
#         shape = np.array(self._knots.shape)
#         coeffs = []
#         for ijk in ndindex:
#             offbounds = (zero > ijk) | (shape <= ijk)
#             coeffs.append(
#                 self._coeffs[ijk] if not np.any(offbounds)
#                 else [0.0] * self.ndim)

#         # coords[:3] += weights.dot(np.array(coeffs, dtype=float))
#         return self.reference.inverse.dot(np.hstack((coords, 1)))[:3]

#     def _map_voxel(self, index, moving=None):
#         """Apply ijk' = f_ijk((i, j, k)), equivalent to the above with indexes."""
#         return tuple(self._moving[index + self.__s])
