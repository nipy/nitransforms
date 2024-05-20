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
    _as_homogeneous,
)
from scipy.ndimage import map_coordinates


class DenseFieldTransform(TransformBase):
    """Represents dense field (voxel-wise) transforms."""

    __slots__ = ("_field", "_deltas")

    def __init__(self, field=None, is_deltas=True, reference=None):
        """
        Create a dense field transform.

        Converting to a field of deformations is straightforward by just adding the corresponding
        displacement to the :math:`(x, y, z)` coordinates of each voxel.
        Numerically, deformation fields are less susceptible to rounding errors
        than displacements fields.
        SPM generally prefers deformations for that reason.

        Parameters
        ----------
        field : :obj:`numpy.array_like` or :obj:`nibabel.SpatialImage`
            The field of deformations or displacements (*deltas*). If given as a data array,
            then the reference **must** be given.
        is_deltas : :obj:`bool`
            Whether this is a displacements (deltas) field (default), or deformations.
        reference : :obj:`ImageGrid`
            Defines the domain of the transform. If not provided, the domain is defined from
            the ``field`` input.

        Example
        -------
        >>> DenseFieldTransform(test_dir / "someones_displacement_field.nii.gz")
        <DenseFieldTransform[3D] (57, 67, 56)>

        """
        if field is None and reference is None:
            raise TransformError("DenseFieldTransforms require a spatial reference")

        super().__init__()

        if field is not None:
            field = _ensure_image(field)
            self._field = np.squeeze(
                np.asanyarray(field.dataobj) if hasattr(field, "dataobj") else field
            )
        else:
            self._field = np.zeros((*reference.shape, reference.ndim), dtype="float32")
            is_deltas = True

        try:
            self.reference = ImageGrid(reference if reference is not None else field)
        except AttributeError:
            raise TransformError(
                "Field must be a spatial image if reference is not provided"
                if reference is None
                else "Reference is not a spatial image"
            )

        if self._field.shape[-1] != self.ndim:
            raise TransformError(
                "The number of components of the field (%d) does not match "
                "the number of dimensions (%d)" % (self._field.shape[-1], self.ndim)
            )

        if is_deltas:
            self._deltas = self._field
            # Convert from displacements (deltas) to deformations fields
            # (just add its origin to each delta vector)
            self._field += self.reference.ndcoords.T.reshape(self._field.shape)

    def __repr__(self):
        """Beautify the python representation."""
        return f"<{self.__class__.__name__}[{self._field.shape[-1]}D] {self._field.shape[:3]}>"

    @property
    def ndim(self):
        """Get the dimensions of the transform."""
        return self._field.ndim - 1

    def map(self, x, inverse=False):
        r"""
        Apply the transformation to a list of physical coordinate points.

        .. math::
            \mathbf{y} = \mathbf{x} + \Delta(\mathbf{x}),
            \label{eq:2}\tag{2}

        where :math:`\Delta(\mathbf{x})` is the value of the discrete field of displacements
        :math:`\Delta` interpolated at the location :math:`\mathbf{x}`.

        Parameters
        ----------
        x : N x D :obj:`numpy.array_like`
            Input RAS+ coordinates (i.e., physical coordinates).
        inverse : :obj:`bool`
            If ``True``, apply the inverse transform :math:`x = f^{-1}(y)`.

        Returns
        -------
        y : N x D :obj:`numpy.array_like`
            Transformed (mapped) RAS+ coordinates (i.e., physical coordinates).

        Examples
        --------
        >>> xfm = DenseFieldTransform(
        ...     test_dir / "someones_displacement_field.nii.gz",
        ...     is_deltas=False,
        ... )
        >>> xfm.map([-6.5, -36., -19.5]).tolist()
        [[0.0, -0.47516798973083496, 0.0]]

        >>> xfm.map([[-6.5, -36., -19.5], [-1., -41.5, -11.25]]).tolist()
        [[0.0, -0.47516798973083496, 0.0], [0.0, -0.538356602191925, 0.0]]

        >>> np.array_str(
        ...     xfm.map([[-6.7, -36.3, -19.2], [-1., -41.5, -11.25]]),
        ...     precision=3,
        ...     suppress_small=True,
        ... )
        '[[ 0.    -0.482  0.   ]\n [ 0.    -0.538  0.   ]]'

        >>> xfm = DenseFieldTransform(
        ...     test_dir / "someones_displacement_field.nii.gz",
        ...     is_deltas=True,
        ... )
        >>> xfm.map([[-6.5, -36., -19.5], [-1., -41.5, -11.25]]).tolist()
        [[-6.5, -36.47516632080078, -19.5], [-1.0, -42.03835678100586, -11.25]]

        >>> np.array_str(
        ...     xfm.map([[-6.7, -36.3, -19.2], [-1., -41.5, -11.25]]),
        ...     precision=3,
        ...     suppress_small=True,
        ... )
        '[[ -6.7   -36.782 -19.2  ]\n [ -1.    -42.038 -11.25 ]]'

        """

        if inverse is True:
            raise NotImplementedError

        ijk = self.reference.index(x)
        indexes = np.round(ijk).astype("int")

        if np.all(np.abs(ijk - indexes) < 1e-3):
            indexes = tuple(tuple(i) for i in indexes)
            return self._field[indexes]

        new_map = np.vstack(
            tuple(
                map_coordinates(
                    self._field[..., i],
                    ijk,
                    order=3,
                    mode="constant",
                    cval=np.nan,
                    prefilter=True,
                )
                for i in range(self.reference.ndim)
            )
        ).T

        # Set NaN values back to the original coordinates value = no displacement
        new_map[np.isnan(new_map)] = np.array(x)[np.isnan(new_map)]
        return new_map

    def __matmul__(self, b):
        """
        Compose with a transform on the right.

        Examples
        --------
        >>> deff = DenseFieldTransform(
        ...     test_dir / "someones_displacement_field.nii.gz",
        ...     is_deltas=False,
        ... )
        >>> deff2 = deff @ TransformBase()
        >>> deff == deff2
        True

        >>> disp = DenseFieldTransform(test_dir / "someones_displacement_field.nii.gz")
        >>> disp2 = disp @ TransformBase()
        >>> disp == disp2
        True

        """
        retval = b.map(self._field.reshape((-1, self._field.shape[-1]))).reshape(
            self._field.shape
        )
        return DenseFieldTransform(retval, is_deltas=False, reference=self.reference)

    def __eq__(self, other):
        """
        Overload equals operator.

        Examples
        --------
        >>> xfm1 = DenseFieldTransform(test_dir / "someones_displacement_field.nii.gz")
        >>> xfm2 = DenseFieldTransform(test_dir / "someones_displacement_field.nii.gz")
        >>> xfm1 == xfm2
        True

        """
        _eq = np.array_equal(self._field, other._field)
        if _eq and self._reference != other._reference:
            warnings.warn("Fields are equal, but references do not match.")
        return _eq

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


load = DenseFieldTransform.from_filename


class BSplineFieldTransform(TransformBase):
    """Represent a nonlinear transform parameterized by BSpline basis."""

    __slots__ = ["_coeffs", "_knots", "_weights", "_order", "_moving"]

    def __init__(self, coefficients, reference=None, order=3):
        """Create a smooth deformation field using B-Spline basis."""
        super().__init__()
        self._order = order

        coefficients = _ensure_image(coefficients)

        self._coeffs = np.asanyarray(coefficients.dataobj)
        self._knots = ImageGrid(coefficients)
        self._weights = None
        if reference is not None:
            self.reference = reference

            if coefficients.shape[-1] != self.reference.ndim:
                raise TransformError(
                    "Number of components of the coefficients does "
                    "not match the number of dimensions"
                )

    @property
    def ndim(self):
        """Get the dimensions of the transform."""
        return self._coeffs.ndim - 1

    def to_field(self, reference=None, dtype="float32"):
        """Generate a displacements deformation field from this B-Spline field."""
        _ref = (
            self.reference if reference is None else ImageGrid(_ensure_image(reference))
        )
        if _ref is None:
            raise TransformError("A reference must be defined")

        if self._weights is None:
            self._weights = grid_bspline_weights(_ref, self._knots)

        field = np.zeros((_ref.npoints, self.ndim))

        for d in range(self.ndim):
            #  1 x Nvox :                          (1 x K) @ (K x Nvox)
            field[:, d] = self._coeffs[..., d].reshape(-1) @ self._weights

        return DenseFieldTransform(
            field.astype(dtype).reshape(*_ref.shape, -1), reference=_ref
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
        >>> xfm.map([-6.5, -36., -19.5]).tolist()  # doctest: +ELLIPSIS
        [[-6.5, -31.476097418406..., -19.5]]

        >>> xfm.map([[-6.5, -36., -19.5], [-1., -41.5, -11.25]]).tolist()  # doctest: +ELLIPSIS
        [[-6.5, -31.4760974184..., -19.5], [-1.0, -3.807267537712..., -11.25]]

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
    nonzero_knots = tuple(
        [np.arange(start, end + 1) for start, end in zip(w_start, w_end)]
    )
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
