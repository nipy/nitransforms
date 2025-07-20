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
from collections import namedtuple
import numpy as np
from pathlib import Path

from nibabel.affines import from_matvec

# Avoids circular imports
try:
    from nitransforms._version import __version__
except ModuleNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

from nitransforms.base import (
    ImageGrid,
    TransformBase,
    _as_homogeneous,
    EQUALITY_TOL,
)
from nitransforms.io import get_linear_factory, TransformFileError
from nitransforms.io.x5 import (
    X5Transform,
    X5Domain,
    to_filename as save_x5,
    from_filename as load_x5,
)


class Affine(TransformBase):
    """Represents linear transforms on image data."""

    __slots__ = ("_matrix", "_inverse")

    def __init__(self, matrix=None, reference=None):
        """
        Initialize a linear transform.

        Parameters
        ----------
        matrix : ndarray
            The coordinate transformation matrix **in physical
            coordinates**, mapping coordinates from *reference* space
            into *moving* space.
            This matrix should be provided in homogeneous coordinates.

        Examples
        --------
        >>> xfm = Affine(reference=test_dir / "someones_anatomy.nii.gz")
        >>> xfm.matrix  # doctest: +NORMALIZE_WHITESPACE
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])

        >>> xfm = Affine([[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        >>> xfm.matrix  # doctest: +NORMALIZE_WHITESPACE
        array([[1, 0, 0, 4],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

        """
        super().__init__(reference=reference)
        self._matrix = np.eye(4)
        self._inverse = np.eye(4)

        if matrix is not None:
            matrix = np.array(matrix)
            if matrix.ndim != 2:
                raise TypeError("Affine should be 2D.")
            elif matrix.shape[0] != matrix.shape[1]:
                raise TypeError("Matrix is not square.")
            self._matrix = matrix

            if not np.allclose(self._matrix[3, :], (0, 0, 0, 1)):
                raise ValueError(
                    """The last row of a homogeneus matrix \
should be (0, 0, 0, 1), got %s."""
                    % self._matrix[3, :]
                )

            # Normalize last row
            self._matrix[3, :] = (0, 0, 0, 1)
            self._inverse = np.linalg.inv(self._matrix)

    def __repr__(self):
        """
        Change representation to the internal matrix.

        Example
        -------
        >>> Affine([
        ...     [1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
        ... ])  # doctest: +NORMALIZE_WHITESPACE
        array([[1, 0, 0, 4],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

        """
        return repr(self.matrix)

    def __eq__(self, other):
        """
        Overload equals operator.

        Examples
        --------
        >>> xfm1 = Affine([[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        >>> xfm2 = Affine(xfm1.matrix)
        >>> xfm1 == xfm2
        True
        >>> xfm1 == Affine()
        False
        >>> xfm1 == TransformBase()
        False

        """
        if not hasattr(other, "matrix"):
            return False

        _eq = np.allclose(self.matrix, other.matrix, rtol=EQUALITY_TOL)
        if _eq and self._reference != other._reference:
            warnings.warn("Affines are equal, but references do not match.")
        return _eq

    def __invert__(self):
        """
        Get the inverse of this transform.

        Example
        -------
        >>> matrix = [[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        >>> Affine(np.linalg.inv(matrix)) == ~Affine(matrix)
        True

        """
        return self.__class__(self._inverse)

    def __len__(self):
        """Enable using len()."""
        return 1 if self._matrix.ndim == 2 else len(self._matrix)

    def __matmul__(self, b):
        """
        Compose two Affines.

        Example
        -------
        >>> xfm1 = Affine([[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        >>> xfm1 @ ~xfm1 == Affine()
        True

        >>> xfm1 = Affine([[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        >>> xfm1 @ np.eye(4) == xfm1
        True

        """
        if not isinstance(b, self.__class__):
            _b = self.__class__(b)
        else:
            _b = b

        retval = self.__class__(self.matrix.dot(_b.matrix))
        if _b.reference:
            retval.reference = _b.reference
        return retval

    @property
    def matrix(self):
        """Access the internal representation of this affine."""
        return self._matrix

    @property
    def ndim(self):
        """Access the internal representation of this affine."""
        return self._matrix.ndim + 1

    @classmethod
    def from_filename(
        cls, filename, fmt=None, reference=None, moving=None, x5_position=0
    ):
        """Create an affine from a transform file."""

        if fmt and fmt.upper() == "X5":
            return from_x5(
                load_x5(filename), reference=reference, x5_position=x5_position
            )

        fmtlist = [fmt] if fmt is not None else ("itk", "lta", "afni", "fsl")

        if fmt is not None and not Path(filename).exists():
            if fmt != "fsl":
                raise FileNotFoundError(
                    f"[Errno 2] No such file or directory: '{filename}'"
                )
            elif not Path(f"{filename}.000").exists():
                raise FileNotFoundError(
                    f"[Errno 2] No such file or directory: '{filename}[.000]'"
                )

        is_array = cls != Affine
        errors = []
        for potential_fmt in fmtlist:
            if potential_fmt == "itk" and Path(filename).suffix == ".mat":
                is_array = False
                cls = Affine

            try:
                struct = get_linear_factory(
                    potential_fmt, is_array=is_array
                ).from_filename(filename)
            except (TransformFileError, FileNotFoundError) as err:
                errors.append((potential_fmt, err))
                continue

            matrix = struct.to_ras(reference=reference, moving=moving)
            return cls(matrix, reference=reference)

        raise TransformFileError(
            f"Could not open <{filename}> (formats tried: {', '.join(fmtlist)})."
        )

    @classmethod
    def from_matvec(cls, mat=None, vec=None, reference=None):
        """
        Create an affine from a matrix and translation pair.

        Example
        -------
        >>> Affine.from_matvec(vec=(4, 0, 0))  # doctest: +NORMALIZE_WHITESPACE
        array([[1., 0., 0., 4.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])

        """
        mat = mat if mat is not None else np.eye(3)
        vec = vec if vec is not None else np.zeros((3,))
        return cls(from_matvec(mat, vector=vec), reference=reference)

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
        >>> xfm = Affine([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
        >>> xfm.map((0,0,0))
        array([[1., 2., 3.]])

        >>> xfm.map((0,0,0), inverse=True)
        array([[-1., -2., -3.]])

        """
        affine = self._matrix
        coords = _as_homogeneous(x, dim=affine.shape[0] - 1).T
        if inverse is True:
            affine = self._inverse
        return affine.dot(coords).T[..., :-1]

    def to_filename(self, filename, fmt="X5", moving=None, x5_inverse=False):
        """Store the transform in the requested output format."""
        if fmt.upper() == "X5":
            return save_x5(filename, [self.to_x5(store_inverse=x5_inverse)])

        writer = get_linear_factory(
            fmt, is_array=isinstance(self, LinearTransformsMapping)
        )

        if fmt.lower() in ("itk", "ants", "elastix"):
            writer.from_ras(self.matrix).to_filename(filename)
        else:
            # Rest of the formats peek into moving and reference image grids
            writer.from_ras(
                self.matrix,
                reference=self.reference,
                moving=ImageGrid(moving) if moving is not None else self.reference,
            ).to_filename(filename)
        return filename

    def to_x5(self, store_inverse=False, metadata=None):
        """Return an :class:`~nitransforms.io.x5.X5Transform` representation."""
        metadata = {"WrittenBy": f"NiTransforms {__version__}"} | (metadata or {})

        domain = None
        if (reference := self.reference) is not None:
            domain = X5Domain(
                grid=True,
                size=getattr(reference or {}, "shape", (0, 0, 0)),
                mapping=reference.affine,
                coordinates="cartesian",
            )
        kinds = tuple("space" for _ in range(self.ndim)) + ("vector",)
        return X5Transform(
            type="linear",
            subtype="affine",
            representation="matrix",
            metadata=metadata,
            transform=self.matrix,
            dimension_kinds=kinds,
            domain=domain,
            inverse=(~self).matrix if store_inverse else None,
            array_length=len(self),
        )


class LinearTransformsMapping(Affine):
    """Represents a series of linear transforms."""

    def __init__(self, transforms, reference=None):
        """
        Initialize a linear transform mapping.

        Parameters
        ----------
        transforms : :obj:`list`
            The inverse coordinate transformation matrix **in physical
            coordinates**, mapping coordinates from *reference* space
            into *moving* space.
            This matrix should be provided in homogeneous coordinates.

        Examples
        --------
        >>> xfm = LinearTransformsMapping([
        ...     [[1., 0, 0, 1.], [0, 1., 0, 2.], [0, 0, 1., 3.], [0, 0, 0, 1.]],
        ...     [[1., 0, 0, -1.], [0, 1., 0, -2.], [0, 0, 1., -3.], [0, 0, 0, 1.]],
        ... ])
        >>> xfm[0].matrix  # doctest: +NORMALIZE_WHITESPACE
        array([[1., 0., 0., 1.],
               [0., 1., 0., 2.],
               [0., 0., 1., 3.],
               [0., 0., 0., 1.]])
        >>> (~xfm)[0].matrix  # doctest: +NORMALIZE_WHITESPACE
        array([[ 1., 0., 0., -1.],
               [ 0., 1., 0., -2.],
               [ 0., 0., 1., -3.],
               [ 0., 0., 0.,  1.]])

        """
        super().__init__(reference=reference)

        self._matrix = np.stack(
            [
                (xfm if isinstance(xfm, Affine) else Affine(xfm)).matrix
                for xfm in transforms
            ],
            axis=0,
        )
        self._inverse = np.linalg.inv(self._matrix)

    def __getitem__(self, i):
        """Enable indexed access to the series of matrices."""
        return Affine(self.matrix[i, ...], reference=self._reference)

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
        >>> xfm = LinearTransformsMapping([
        ...     [[1., 0, 0, 1.], [0, 1., 0, 2.], [0, 0, 1., 3.], [0, 0, 0, 1.]],
        ...     [[1., 0, 0, -1.], [0, 1., 0, -2.], [0, 0, 1., -3.], [0, 0, 0, 1.]],
        ... ])
        >>> xfm.matrix
        array([[[ 1.,  0.,  0.,  1.],
                [ 0.,  1.,  0.,  2.],
                [ 0.,  0.,  1.,  3.],
                [ 0.,  0.,  0.,  1.]],
        <BLANKLINE>
               [[ 1.,  0.,  0., -1.],
                [ 0.,  1.,  0., -2.],
                [ 0.,  0.,  1., -3.],
                [ 0.,  0.,  0.,  1.]]])

        >>> y = xfm.map([(0, 0, 0), (-1, -1, -1), (1, 1, 1)])
        >>> y[0, :, :3]
        array([[1., 2., 3.],
               [0., 1., 2.],
               [2., 3., 4.]])

        >>> y = xfm.map([(0, 0, 0), (-1, -1, -1), (1, 1, 1)], inverse=True)
        >>> y[0, :, :3]
        array([[-1., -2., -3.],
               [-2., -3., -4.],
               [ 0., -1., -2.]])


        """
        affine = self.matrix
        coords = _as_homogeneous(x, dim=affine.shape[-1] - 1).T
        if inverse is True:
            affine = self._inverse
        return np.swapaxes(affine.dot(coords), 1, 2)


def load(filename, fmt=None, reference=None, moving=None):
    """
    Load a linear transform file.

    Examples
    --------
    >>> xfm = load(regress_dir / "affine-LAS.itk.tfm")
    >>> isinstance(xfm, Affine)
    True

    >>> xfm = load(regress_dir / "itktflist.tfm")
    >>> isinstance(xfm, LinearTransformsMapping)
    True

    """
    xfm = LinearTransformsMapping.from_filename(
        filename, fmt=fmt, reference=reference, moving=moving
    )

    if isinstance(xfm, LinearTransformsMapping) and len(xfm) == 1:
        xfm = xfm[0]

    return xfm


def from_x5(x5_list, reference=None, x5_position=0):
    """Create an affine from a list of :class:`~nitransforms.io.x5.X5Transform` objects."""

    x5_xfm = x5_list[x5_position]
    Transform = Affine if x5_xfm.array_length == 1 else LinearTransformsMapping
    if (
        x5_xfm.domain and not x5_xfm.domain.grid and len(x5_xfm.domain.size) == 3
    ):  # pragma: no cover
        raise NotImplementedError("Only 3D regularly gridded domains are supported")
    elif x5_xfm.domain:
        # Override reference
        Domain = namedtuple("Domain", "affine shape")
        reference = Domain(x5_xfm.domain.mapping, x5_xfm.domain.size)

    return Transform(x5_xfm.transform, reference=reference)
