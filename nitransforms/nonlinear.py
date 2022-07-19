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
import numpy as np

from nitransforms import io
from nitransforms.io.base import _ensure_image
from nitransforms.interp.bspline import grid_bspline_weights, _cubic_bspline
from nitransforms.base import (
    TransformBase,
    TransformError,
    ImageGrid,
    SpatialReference,
    _as_homogeneous,
    EQUALITY_TOL,
)


class DeformationFieldTransform(TransformBase):
    """Represents a dense field of deformed locations (corresponding to each voxel)."""

    __slots__ = ["_field"]

    def __init__(self, field, reference=None):
        """
        Create a dense deformation field transform.

        Example
        -------
        >>> DeformationFieldTransform(test_dir / "someones_displacement_field.nii.gz")
        <DeformationFieldTransform[3D] (57, 67, 56)>

        """
        super().__init__()

        field = _ensure_image(field)
        self._field = np.squeeze(
            np.asanyarray(field.dataobj) if hasattr(field, "dataobj") else field
        )

        try:
            self.reference = ImageGrid(
                reference if reference is not None else field
            )
        except AttributeError:
            raise TransformError(
                "Field must be a spatial image if reference is not provided"
                if reference is None else
                "Reference is not a spatial image"
            )

        ndim = self._field.ndim - 1
        if self._field.shape[-1] != ndim:
            raise TransformError(
                "The number of components of the displacements (%d) does not match "
                "the number of dimensions (%d)" % (self._field.shape[-1], ndim)
            )

    def __repr__(self):
        """Beautify the python representation."""
        return f"<{self.__class__.__name__}[{self._field.shape[-1]}D] {self._field.shape[:3]}>"

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
        >>> xfm = DeformationFieldTransform(test_dir / "someones_displacement_field.nii.gz")
        >>> xfm.map([-6.5, -36., -19.5]).tolist()
        [[0.0, -0.47516798973083496, 0.0]]

        >>> xfm.map([[-6.5, -36., -19.5], [-1., -41.5, -11.25]]).tolist()
        [[0.0, -0.47516798973083496, 0.0], [0.0, -0.538356602191925, 0.0]]

        """

        if inverse is True:
            raise NotImplementedError
        ijk = self.reference.index(x)
        indexes = np.round(ijk).astype("int")
        if np.any(np.abs(ijk - indexes) > 0.05):
            warnings.warn("Some coordinates are off-grid of the displacements field.")
        indexes = tuple(tuple(i) for i in indexes.T)
        return self._field[indexes]

    def __matmul__(self, b):
        """
        Compose with a transform on the right.

        Examples
        --------
        >>> xfm = DeformationFieldTransform(test_dir / "someones_displacement_field.nii.gz")
        >>> xfm2 = xfm @ TransformBase()
        >>> xfm == xfm2
        True

        """
        retval = b.map(
            self._field.reshape((-1, self._field.shape[-1]))
        ).reshape(self._field.shape)
        return DeformationFieldTransform(retval, reference=self.reference)

    def __eq__(self, other):
        """
        Overload equals operator.

        Examples
        --------
        >>> xfm1 = DeformationFieldTransform(test_dir / "someones_displacement_field.nii.gz")
        >>> xfm2 = DeformationFieldTransform(test_dir / "someones_displacement_field.nii.gz")
        >>> xfm1 == xfm2
        True

        """
        _eq = np.array_equal(self._field, other._field)
        if _eq and self._reference != other._reference:
            warnings.warn("Fields are equal, but references do not match.")
        return _eq


class DisplacementsFieldTransform(DeformationFieldTransform):
    """
    Represents a dense field of displacements (one vector per voxel).

    Converting to a field of deformations is straightforward by just adding the corresponding
    displacement to the :math:`(x, y, z)` coordinates of each voxel.
    Numerically, deformation fields are less susceptible to rounding errors
    than displacements fields.
    SPM generally prefers deformations for that reason.

    """

    __slots__ = ["_displacements"]

    def __init__(self, field, reference=None):
        """
        Create a transform supported by a field of voxel-wise displacements.

        Example
        -------
        >>> xfm = DisplacementsFieldTransform(test_dir / "someones_displacement_field.nii.gz")
        >>> xfm
        <DisplacementsFieldTransform[3D] (57, 67, 56)>

        >>> xfm.map([[-6.5, -36., -19.5], [-1., -41.5, -11.25]]).tolist()
        [[-6.5, -36.47516632080078, -19.5], [-1.0, -42.03835678100586, -11.25]]

        """
        super().__init__(field, reference=reference)
        self._displacements = self._field
        # Convert from displacements to deformations fields
        # (just add the origin to the displacements vector)
        self._field += self.reference.ndcoords.T.reshape(self._field.shape)

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

    def __init__(self, coefficients, reference=None, order=3):
        """Create a smooth deformation field using B-Spline basis."""
        super(BSplineFieldTransform, self).__init__()
        self._order = order

        coefficients = _ensure_image(coefficients)

        self._coeffs = np.asanyarray(coefficients.dataobj)
        self._knots = ImageGrid(coefficients)
        self._weights = None
        if reference is not None:
            self.reference = reference

            if coefficients.shape[-1] != self.ndim:
                raise TransformError(
                    'Number of components of the coefficients does '
                    'not match the number of dimensions')

    def to_field(self, reference=None, dtype="float32"):
        """Generate a displacements deformation field from this B-Spline field."""
        _ref = (
            self.reference if reference is None else
            ImageGrid(_ensure_image(reference))
        )
        if _ref is None:
            raise TransformError("A reference must be defined")

        ndim = self._coeffs.shape[-1]

        if self._weights is None:
            self._weights = grid_bspline_weights(_ref, self._knots)

        field = np.zeros((_ref.npoints, ndim))

        for d in range(ndim):
            #  1 x Nvox :                          (1 x K) @ (K x Nvox)
            field[:, d] = self._coeffs[..., d].reshape(-1) @ self._weights

        return DisplacementsFieldTransform(
            field.astype(dtype).reshape(*_ref.shape, -1), reference=_ref)

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

        _ref = (
            self.reference if reference is None else
            SpatialReference.factory(_ensure_image(reference))
        )
        spatialimage = _ensure_image(spatialimage)

        # If locations to be interpolated are not on a grid, run map()
        if not isinstance(_ref, ImageGrid):
            return super().apply(
                spatialimage,
                reference=_ref,
                order=order,
                mode=mode,
                cval=cval,
                prefilter=prefilter,
                output_dtype=output_dtype,
            )

        # If locations to be interpolated are on a grid, generate a displacements field
        return self.to_field(reference=reference).apply(
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
            \label{eq:1}\tag{1}

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
        >>> xfm = BSplineFieldTransform(test_dir / "someones_bspline_coefficients.nii.gz")
        >>> xfm.reference = test_dir / "someones_anatomy.nii.gz"
        >>> xfm.map([-6.5, -36., -19.5]).tolist()
        [[-6.5, -31.476097418406784, -19.5]]

        >>> xfm.map([[-6.5, -36., -19.5], [-1., -41.5, -11.25]]).tolist()
        [[-6.5, -31.476097418406784, -19.5], [-1.0, -3.8072675377121996, -11.25]]

        """
        vfunc = partial(
            _map_xyz,
            reference=self.reference,
            knots=self._knots,
            coeffs=self._coeffs,
        )
        return np.array([vfunc(_x).tolist() for _x in np.atleast_2d(x)])


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
