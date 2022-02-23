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
from functools import partial
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from nibabel.funcs import four_to_three
from nibabel.loadsave import load as _nbload

from nitransforms import io
from nitransforms.io.base import _ensure_image
from nitransforms.interp.bspline import grid_bspline_weights, _cubic_bspline
from nitransforms.base import (
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

        self._field = np.asanyarray(field.dataobj) if hasattr(field, "dataobj") else field
        self.reference = reference or field.__class__(
            np.zeros(self._field.shape[:-1]), field.affine, field.header
        )

        ndim = self._field.ndim - 1
        if self._field.shape[-1] != ndim:
            raise ValueError(
                "The number of components of the displacements (%d) does not "
                "the number of dimensions (%d)" % (self._field.shape[-1], ndim)
            )

    def map(self, x, inverse=False):
        r"""
        Apply the transformation to a list of physical coordinate points.

        .. math::
            \mathbf{y} = \mathbf{x} + D(\mathbf{x}),
            \label{eq:2}\tag{2}

        where :math:`D(\mathbf{x})` is the value of the discrete field of displacements
        :math:`D` interpolated at the location :math:`\mathbf{x}`.

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

    def __init__(self, reference, coefficients, order=3):
        """Create a smooth deformation field using B-Spline basis."""
        super(BSplineFieldTransform, self).__init__()
        self._order = order
        self.reference = reference

        coefficients = _ensure_image(coefficients)
        if coefficients.shape[-1] != self.ndim:
            raise ValueError(
                'Number of components of the coefficients does '
                'not match the number of dimensions')

        self._coeffs = np.asanyarray(coefficients.dataobj)
        self._knots = ImageGrid(four_to_three(coefficients)[0])
        self._weights = None

    def to_field(self, reference=None):
        """Generate a displacements deformation field from this B-Spline field."""
        reference = _ensure_image(reference)
        _ref = self.reference if reference is None else SpatialReference.factory(reference)
        ndim = self._coeffs.shape[-1]

        # If locations to be interpolated are on a grid, use faster tensor-bspline calculation
        if self._weights is None:
            self._weights = grid_bspline_weights(_ref, self._knots)

        field = np.zeros((_ref.npoints, ndim))

        for d in range(ndim):
            field[:, d] = self._coeffs[..., d].reshape(-1) @ self._weights

        return field.astype("float32")

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

        if reference is not None:
            reference = _ensure_image(reference)

        _ref = (
            self.reference if reference is None else SpatialReference.factory(reference)
        )

        if isinstance(spatialimage, (str, Path)):
            spatialimage = _nbload(str(spatialimage))

        # If locations to be interpolated are not on a grid, run map()
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

        # If locations to be interpolated are on a grid, generate a displacements field
        return DisplacementsFieldTransform(
            self.to_field().reshape((*(_ref.shape), -1)),
            reference=_ref,
        ).apply(
            spatialimage,
            reference=reference,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
            output_dtype=output_dtype,
        )

    def map(self, x, inverse=False):
        r"""
        Apply the transformation to a list of physical coordinate points.

        .. math::
            \mathbf{y} = \mathbf{x} + \Psi^3(\mathbf{k}, \mathbf{x}),
            \label{eq:2}\tag{2}

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
        vfunc = partial(
            _map_xyz,
            reference=self.reference,
            knots=self._knots,
            coeffs=self._coeffs,
        )
        return [vfunc(_x) for _x in np.atleast_2d(x)]


def _map_xyz(x, reference, knots, coeffs):
    """Apply the transformation to just one coordinate."""
    ndim = len(x)
    # Calculate the index coordinates of the point in the B-Spline grid
    ijk = (knots.inverse @ _as_homogeneous(x).squeeze())[:ndim]

    # Determine the window within distance 2.0 (where the B-Spline is nonzero)
    # Probably this will change if the order of the B-Spline is different
    w_start, w_end = np.ceil(ijk - 2).astype(int), np.floor(ijk + 2).astype(int)
    # Generate a grid of indexes corresponding to the window
    nonzero_knots = tuple([
        np.arange(start, end + 1) for start, end in zip(w_start, w_end)
    ])
    nonzero_knots = tuple(np.meshgrid(*nonzero_knots, indexing="ij"))
    window = np.array(nonzero_knots).reshape((ndim, -1))

    # Calculate the distance of the location w.r.t. to all voxels in window
    distance = window.T - ijk
    # Since this is a grid, distance only takes a few float values
    unique_d, indices = np.unique(distance.reshape(-1), return_inverse=True)
    # Calculate the B-Spline weight corresponding to the distance.
    # Then multiply the three weights of each knot (tensor-product B-Spline)
    tensor_bspline = _cubic_bspline(unique_d)[indices].reshape(distance.shape).prod(1)
    # Extract the values of the coefficients in the window
    coeffs = coeffs[nonzero_knots].reshape((-1, ndim))
    # Inference: the displacement is the product of coefficients x tensor-product B-Splines
    return x + coeffs.T @ tensor_bspline
