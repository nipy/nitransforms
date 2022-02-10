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
from pathlib import Path
import numpy as np
from scipy.sparse import vstack as sparse_vstack
from scipy import ndimage as ndi
from nibabel.funcs import four_to_three
from nibabel.loadsave import load as _nbload

from . import io
from .interp.bspline import grid_bspline_weights
from .base import (
    TransformBase,
    ImageGrid,
    SpatialReference,
    _as_homogeneous,
)


class DisplacementsFieldTransform(TransformBase):
    """Represents a dense field of displacements (one vector per voxel)."""

    __slots__ = ["_field"]

    def __init__(self, field, reference=None):
        """Create a dense deformation field transform."""
        super().__init__()
        self._field = np.asanyarray(field.dataobj)

        ndim = self._field.ndim - 1
        if self._field.shape[-1] != ndim:
            raise ValueError(
                "The number of components of the displacements (%d) does not "
                "the number of dimensions (%d)" % (self._field.shape[-1], ndim)
            )

        self.reference = field.__class__(
            np.zeros(self._field.shape[:-1]), field.affine, field.header
        )

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
        indexes = np.round(ijk).astype("int")
        if np.any(np.abs(ijk - indexes) > 0.05):
            warnings.warn("Some coordinates are off-grid of the displacements field.")
        indexes = tuple(tuple(i) for i in indexes.T)
        return x + self._field[indexes]

    @classmethod
    def from_filename(cls, filename, fmt="X5"):
        _factory = {
            "afni": io.afni.AFNIDisplacementsField,
            "itk": io.itk.ITKDisplacementsField,
            "fsl": io.fsl.FSLDisplacementsField,
        }
        if fmt not in _factory:
            raise NotImplementedError(f"Unsupported format <{fmt}>")

        return cls(_factory[fmt].from_filename(filename))


load = DisplacementsFieldTransform.from_filename


class BSplineFieldTransform(TransformBase):
    """Represent a nonlinear transform parameterized by BSpline basis."""

    __slots__ = ['_coeffs', '_knots', '_weights', '_order', '_moving']
    __s = (slice(None), )

    def __init__(self, reference, coefficients, order=3):
        """Create a smooth deformation field using B-Spline basis."""
        super(BSplineFieldTransform, self).__init__()
        self._order = order
        self.reference = reference

        if coefficients.shape[-1] != self.ndim:
            raise ValueError(
                'Number of components of the coefficients does '
                'not match the number of dimensions')

        self._coeffs = np.asanyarray(coefficients.dataobj)
        self._knots = ImageGrid(four_to_three(coefficients)[0])
        self._weights = None

    def apply(
        self,
        spatialimage,
        reference=None,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
        output_dtype=None,
    ):
        """Apply a B-Spline transform on input data."""

        if reference is not None and isinstance(reference, (str, Path)):
            reference = _nbload(str(reference))

        _ref = (
            self.reference if reference is None else SpatialReference.factory(reference)
        )

        if isinstance(spatialimage, (str, Path)):
            spatialimage = _nbload(str(spatialimage))

        if not isinstance(_ref, ImageGrid):
            return super().apply(
                spatialimage,
                reference=reference,
                order=order,
                mode=mode,
                cval=cval,
                prefilter=prefilter,
                output_dtype=output_dtype,
            )

        # If locations to be interpolated are on a grid, use faster tensor-bspline calculation
        if self._weights is None:
            self._weights = grid_bspline_weights(_ref, self._knots)

        ycoords = _ref.ndcoords.T + (
            np.squeeze(np.hstack(self._coeffs).T) @ sparse_vstack(self._weights)
        )

        data = np.squeeze(np.asanyarray(spatialimage.dataobj))
        output_dtype = output_dtype or data.dtype
        targets = ImageGrid(spatialimage).index(  # data should be an image
            _as_homogeneous(np.vstack(ycoords), dim=_ref.ndim)
        )

        if data.ndim == 4:
            if len(self) != data.shape[-1]:
                raise ValueError(
                    "Attempting to apply %d transforms on a file with "
                    "%d timepoints" % (len(self), data.shape[-1])
                )
            targets = targets.reshape((len(self), -1, targets.shape[-1]))
            resampled = np.stack(
                [
                    ndi.map_coordinates(
                        data[..., t],
                        targets[t, ..., : _ref.ndim].T,
                        output=output_dtype,
                        order=order,
                        mode=mode,
                        cval=cval,
                        prefilter=prefilter,
                    )
                    for t in range(data.shape[-1])
                ],
                axis=0,
            )
        elif data.ndim in (2, 3):
            resampled = ndi.map_coordinates(
                data,
                targets[..., : _ref.ndim].T,
                output=output_dtype,
                order=order,
                mode=mode,
                cval=cval,
                prefilter=prefilter,
            )

        newdata = resampled.reshape((len(self), *_ref.shape))
        moved = spatialimage.__class__(
            np.moveaxis(newdata, 0, -1), _ref.affine, spatialimage.header
        )
        moved.header.set_data_dtype(output_dtype)
        return moved

    def map(self, x, inverse=False):
        raise NotImplementedError

    def _map_voxel(self, index, moving=None):
        """Apply ijk' = f_ijk((i, j, k)), equivalent to the above with indexes."""
        raise NotImplementedError
