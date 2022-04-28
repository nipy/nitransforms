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
from pathlib import Path
from scipy import ndimage as ndi

from nibabel.loadsave import load as _nbload

from nitransforms.base import (
    ImageGrid,
    TransformBase,
    SpatialReference,
    _as_homogeneous,
    EQUALITY_TOL,
)
from nitransforms.io import get_linear_factory, TransformFileError


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

    def _to_hdf5(self, x5_root):
        """Serialize this object into the x5 file format."""
        xform = x5_root.create_dataset("Transform", data=[self._matrix])
        xform.attrs["Type"] = "affine"
        x5_root.create_dataset("Inverse", data=[(~self).matrix])

        if self._reference:
            self.reference._to_hdf5(x5_root.create_group("Reference"))

    def to_filename(self, filename, fmt="X5", moving=None):
        """Store the transform in the requested output format."""
        writer = get_linear_factory(fmt, is_array=False)

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

    @classmethod
    def from_filename(cls, filename, fmt=None, reference=None, moving=None):
        """Create an affine from a transform file."""
        fmtlist = [fmt] if fmt is not None else ("itk", "lta", "afni", "fsl")

        for potential_fmt in fmtlist:
            try:
                struct = get_linear_factory(potential_fmt).from_filename(filename)
                matrix = struct.to_ras(reference=reference, moving=moving)
                if cls == Affine:
                    if np.shape(matrix)[0] != 1:
                        raise TypeError("Cannot load transform array '%s'" % filename)
                    matrix = matrix[0]
                return cls(matrix, reference=reference)
            except (TransformFileError, FileNotFoundError):
                continue

        raise TransformFileError(
            f"Could not open <{filename}> (formats tried: {', '.join(fmtlist)})."
        )

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

    def __len__(self):
        """Enable using len()."""
        return len(self._matrix)

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

    def to_filename(self, filename, fmt="X5", moving=None):
        """Store the transform in the requested output format."""
        writer = get_linear_factory(fmt, is_array=True)

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
        mode : {"constant", "reflect", "nearest", "mirror", "wrap"}, optional
            Determines how the input image is extended when the resamplings overflows
            a border. Default is "constant".
        cval : float, optional
            Constant value for ``mode="constant"``. Default is 0.0.
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

        _ref = (
            self.reference if reference is None else SpatialReference.factory(reference)
        )

        if isinstance(spatialimage, (str, Path)):
            spatialimage = _nbload(str(spatialimage))

        data = np.squeeze(np.asanyarray(spatialimage.dataobj))
        output_dtype = output_dtype or data.dtype

        ycoords = self.map(_ref.ndcoords.T)
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

        if isinstance(_ref, ImageGrid):  # If reference is grid, reshape
            newdata = resampled.reshape((len(self), *_ref.shape))
            moved = spatialimage.__class__(
                np.moveaxis(newdata, 0, -1), _ref.affine, spatialimage.header
            )
            moved.header.set_data_dtype(output_dtype)
            return moved

        return resampled


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
    if len(xfm) == 1:
        return xfm[0]
    return xfm
