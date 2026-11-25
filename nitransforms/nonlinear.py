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
from collections import namedtuple
import numpy as np
import nibabel as nb

from nitransforms import io
from nitransforms.io.base import _ensure_image
from nitransforms.io.x5 import from_filename as load_x5
from nitransforms.base import (
    TransformBase,
    TransformError,
    ImageGrid,
)
from scipy.ndimage import map_coordinates, spline_filter
from scipy.interpolate import NdBSpline

# Avoids circular imports
try:
    from nitransforms._version import __version__
except ModuleNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"


class DenseFieldTransform(TransformBase):
    """Represents dense field (voxel-wise) transforms."""

    __slots__ = ("_field", "_deltas", "_is_deltas", "_prefiltered")

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
            raise TransformError("cannot initialize field")

        super().__init__()

        if field is not None:
            field = _ensure_image(field)
            # Extract data if nibabel object otherwise assume numpy array
            _data = np.squeeze(
                np.asanyarray(field.dataobj)
                if hasattr(field, "dataobj")
                else field.copy()
            )

        try:
            self.reference = ImageGrid(reference if reference is not None else field)
        except AttributeError:
            raise TransformError(
                "field must be a spatial image if reference is not provided"
                if reference is None
                else "reference is not a spatial image"
            )

        fieldshape = (*self.reference.shape, self.reference.ndim)
        if field is None:
            _data = np.zeros(fieldshape)
        elif fieldshape != _data.shape:
            raise TransformError(
                f"Shape of the field ({'x'.join(str(i) for i in _data.shape)}) "
                f"doesn't match that of the reference({'x'.join(str(i) for i in fieldshape)})"
            )

        self._is_deltas = is_deltas
        self._prefiltered = None
        self._field = self.reference.ndcoords.reshape(fieldshape)

        if self.is_deltas:
            self._deltas = _data.copy()
            self._field += self._deltas
        else:
            self._field = _data.copy()

    def __repr__(self):
        """Beautify the python representation."""
        return f"<{self.__class__.__name__}[{self._field.shape[-1]}D] {self._field.shape[:3]}>"

    @property
    def is_deltas(self):
        """Check whether this is a displacements (``True``) or a deformation (``False``) field."""
        return self._is_deltas

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
        >>> xfm.map([[-6.5, -36., -19.5]]).tolist()
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
        >>> xfm.map([[-6.5, -36., -19.5], [-1., -41.5, -11.25]]).tolist()  # doctest: +ELLIPSIS
        [[-6.5, -36.475..., -19.5], [-1.0, -42.038..., -11.25]]

        >>> np.array_str(
        ...     xfm.map([[-6.7, -36.3, -19.2], [-1., -41.5, -11.25]]),
        ...     precision=3,
        ...     suppress_small=True,
        ... )
        '[[ -6.7   -36.782 -19.2  ]\n [ -1.    -42.038 -11.25 ]]'

        """

        if inverse is True:
            raise NotImplementedError

        ijk = self.reference.index(np.array(x, dtype="float32"))
        # Points up to the outermost voxel centers, with the tolerance of the grid check
        inside = np.all(
            (ijk > -1e-3) & (ijk < np.array(self.reference.shape) - 1 + 1e-3), axis=1
        )
        indexes = np.round(ijk).astype("int")
        ongrid = np.where(inside & (np.linalg.norm(ijk - indexes, axis=1) < 1e-3))[0]

        if ongrid.size == np.shape(x)[0]:
            # return self._field[*indexes.T, :]  # From Python 3.11
            return self._field[tuple(indexes.T) + (np.s_[:],)]

        # map_coordinates does not support `mode='reflect', reflect_type='odd'`,
        # so perform our own padding and prefiltering to extend the positions
        # linearly past the edges. 12 is sufficient padding to ensure the spline
        # reproduces the field up to the outermost voxel centers.
        #
        # Cache the prefiltered field to speed up multiple calls.
        pad = 12
        if self._prefiltered is None:
            # Filter once, on first use, rather than on every call
            field = np.pad(
                self._field,
                [(pad, pad)] * self.reference.ndim + [(0, 0)],
                mode="reflect",
                reflect_type="odd",
            )
            self._prefiltered = np.stack(
                [
                    spline_filter(field[..., i], order=3, mode="nearest")
                    for i in range(self.reference.ndim)
                ],
                axis=-1,
            )
        mapped_coords = np.vstack(
            tuple(
                map_coordinates(
                    self._prefiltered[..., i],
                    ijk.T + pad,
                    order=3,
                    mode="nearest",
                    prefilter=False,
                )
                for i in range(self.reference.ndim)
            )
        ).T

        # Leave points beyond the field in place
        mapped_coords[~inside] = np.asarray(x)[~inside]
        return mapped_coords

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
        >>> xfm1 == TransformBase()
        False
        >>> xfm1 == BSplineFieldTransform(test_dir / "someones_bspline_coefficients.nii.gz")
        False

        """
        if not hasattr(other, "_field") or self._field.shape != other._field.shape:
            return False

        _eq = np.allclose(self._field, other._field)
        if _eq and self._reference != other._reference:
            warnings.warn("Fields are equal, but references do not match.")
        return _eq

    def to_filename(self, filename, fmt="X5", moving=None, x5_inverse=False):
        """Store the transform in the designated format."""

        if fmt.upper() == "X5":
            raise TypeError("Please use .to_x5()")

        field = nb.Nifti1Image(
            self._deltas if self.is_deltas else self._field,
            self.reference.affine,
            None,
        )

        if fmt.lower() == "afni":
            from nitransforms.io.afni import AFNIDisplacementsField as FieldIOType

        elif fmt.lower() in ("itk", "ants", "elastix"):
            from nitransforms.io.itk import ITKDisplacementsField as FieldIOType

        elif fmt.lower() == "fsl":
            from nitransforms.io.fsl import FSLDisplacementsField as FieldIOType

        else:
            raise NotImplementedError(
                f"Dense field of type '{fmt}' cannot be converted."
            )

        FieldIOType.to_image(field).to_filename(filename)

    def to_x5(self, metadata=None):
        """Return an :class:`~nitransforms.io.x5.X5Transform` representation."""
        metadata = {"WrittenBy": f"NiTransforms {__version__}"} | (metadata or {})

        domain = None
        if (reference := self.reference) is not None:
            domain = io.x5.X5Domain(
                grid=True,
                size=getattr(reference, "shape", (0, 0, 0)),
                mapping=reference.affine,
                coordinates="cartesian",
            )

        kinds = tuple("space" for _ in range(self.ndim)) + ("vector",)

        return io.x5.X5Transform(
            type="nonlinear",
            subtype="densefield",
            representation="displacements" if self.is_deltas else "deformations",
            metadata=metadata,
            transform=self._deltas if self.is_deltas else self._field,
            dimension_kinds=kinds,
            domain=domain,
        )

    @classmethod
    def from_filename(cls, filename, fmt="X5", x5_position=0):
        _factory = {
            "afni": io.afni.AFNIDisplacementsField,
            "itk": io.itk.ITKDisplacementsField,
            "fsl": io.fsl.FSLDisplacementsField,
            "X5": None,
        }
        fmt = fmt.upper()
        if fmt not in {k.upper() for k in _factory}:
            raise NotImplementedError(f"Unsupported format <{fmt}>")

        if fmt == "X5":
            return from_x5(load_x5(filename), x5_position=x5_position)

        return cls(_factory[fmt.lower()].from_filename(filename))


load = DenseFieldTransform.from_filename


class BSplineFieldTransform(TransformBase):
    """Represent a nonlinear transform parameterized by BSpline basis."""

    __slots__ = ["_coeffs", "_knots", "_spline", "_order", "_moving"]

    def __init__(self, coefficients, reference=None, order=3):
        """Create a smooth deformation field using B-Spline basis."""
        super().__init__()
        self._order = order

        coefficients = _ensure_image(coefficients)

        self._coeffs = np.asanyarray(coefficients.dataobj)
        self._knots = ImageGrid(coefficients)
        # Center each basis function on its knot. order + 1 zero coefficients per
        # side make the outermost polynomial pieces identically zero, so the field
        # is 0 beyond the support of the lattice.
        pad = order + 1
        self._spline = NdBSpline(
            tuple(
                np.arange(-pad, n + pad + order + 1) - (order + 1) / 2
                for n in self._knots.shape
            ),
            np.pad(self._coeffs, [(pad, pad)] * self._knots.ndim + [(0, 0)]),
            order,
        )
        if reference is not None:
            self.reference = reference

            if coefficients.shape[-1] != self.reference.ndim:
                raise TransformError(
                    "Number of components of the coefficients does "
                    "not match the number of dimensions"
                )

    def __eq__(self, other):
        """
        Overload equals operator.

        Examples
        --------
        >>> xfm1 = BSplineFieldTransform(test_dir / "someones_bspline_coefficients.nii.gz")
        >>> xfm2 = BSplineFieldTransform(test_dir / "someones_bspline_coefficients.nii.gz")
        >>> xfm1 == xfm2
        True
        >>> xfm2._coeffs[:, :, :] = 0  # Let's zero all coefficients
        >>> xfm1 == xfm2
        False
        >>> xfm2 = BSplineFieldTransform(
        ...     test_dir / "someones_bspline_coefficients.nii.gz",
        ...     order=4,
        ... )
        >>> xfm1 == xfm2
        False
        >>> xfm1 == TransformBase()
        False
        >>> xfm1 == DenseFieldTransform(test_dir / "someones_displacement_field.nii.gz")
        False

        """
        if not hasattr(other, "_coeffs") or self._coeffs.shape != other._coeffs.shape:
            return False

        _eq = self._order == other._order
        _eq = _eq and np.allclose(self._coeffs, other._coeffs)

        if _eq and self._reference != other._reference:
            warnings.warn("Coefficients are equal, but references do not match.")
        return _eq

    @property
    def ndim(self):
        """Get the dimensions of the transform."""
        return self._coeffs.ndim - 1

    @classmethod
    def from_filename(cls, filename, fmt="X5", x5_position=0):
        _factory = {
            "X5": None,
        }
        fmt = fmt.upper()
        if fmt not in {k.upper() for k in _factory}:
            raise NotImplementedError(f"Unsupported format <{fmt}>")

        return from_x5(load_x5(filename), x5_position=x5_position)
        # return cls(_factory[fmt.lower()].from_filename(filename))

    def to_field(self, reference=None, dtype="float32"):
        """Generate a displacements deformation field from this B-Spline field."""
        _ref = (
            self.reference if reference is None else ImageGrid(_ensure_image(reference))
        )
        if _ref is None:
            raise TransformError("A reference must be defined")

        field = self._spline(self._knots.index(_ref.ndcoords))

        return DenseFieldTransform(
            field.astype(dtype).reshape(*_ref.shape, -1), reference=_ref
        )

    def to_x5(self, metadata=None):
        """Return an :class:`~nitransforms.io.x5.X5Transform` representation."""
        metadata = {"WrittenBy": f"NiTransforms {__version__}"} | (metadata or {})

        domain = None
        if (reference := self.reference) is not None:
            domain = io.x5.X5Domain(
                grid=True,
                size=getattr(reference, "shape", (0, 0, 0)),
                mapping=reference.affine,
                coordinates="cartesian",
            )

        kinds = tuple("space" for _ in range(self.ndim)) + ("vector",)

        return io.x5.X5Transform(
            type="nonlinear",
            subtype="bspline",
            representation="coefficients",
            metadata=metadata,
            transform=self._coeffs,
            dimension_kinds=kinds,
            domain=domain,
            additional_parameters=self._knots.affine,
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
        [[-6.5, -36.475114..., -19.5]]

        >>> xfm.map([[-6.5, -36., -19.5], [-1., -41.5, -11.25]]).tolist()  # doctest: +ELLIPSIS
        [[-6.5, -36.475114..., -19.5], [-1.0, -42.038789..., -11.25]]

        """
        x = np.atleast_2d(x)
        return x + self._spline(self._knots.index(x))


def from_x5(x5_list, x5_position=0):
    """Create a transform from a list of :class:`~nitransforms.io.x5.X5Transform` objects."""

    x5_xfm = x5_list[x5_position]

    Transform = (
        BSplineFieldTransform if x5_xfm.subtype == "bspline" else DenseFieldTransform
    )
    Domain = namedtuple("Domain", "affine shape")
    reference = Domain(x5_xfm.domain.mapping, x5_xfm.domain.size)
    xfm_params = (
        nb.Nifti1Image(x5_xfm.transform, x5_xfm.additional_parameters)
        if x5_xfm.subtype == "bspline"
        else x5_xfm.transform
    )

    xfm_kwargs = (
        {}
        if x5_xfm.subtype == "bspline"
        else {"is_deltas": x5_xfm.representation == "displacements"}
    )

    return Transform(xfm_params, reference=reference, **xfm_kwargs)
