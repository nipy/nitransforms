# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Linear transforms."""
from pathlib import Path
import warnings
import numpy as np

from nibabel.loadsave import load as loadimg
from nibabel.affines import from_matvec, voxel_sizes, obliquity
from .base import TransformBase, _as_homogeneous, EQUALITY_TOL
from .patched import shape_zoom_affine
from . import io


LPS = np.diag([-1, -1, 1, 1])
OBLIQUITY_THRESHOLD_DEG = 0.01


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
            itkobj = io.itk.ITKLinearTransformArray(
                xforms=[LPS.dot(m.dot(LPS)) for m in self.matrix])
            with open(filename, 'w') as f:
                f.write(itkobj.to_string())
            return filename

        if fmt.lower() == 'afni':
            from math import pi

            if moving and isinstance(moving, (str, bytes, Path)):
                moving = loadimg(str(moving))

            T = self.matrix.copy()
            pre = LPS
            post = LPS
            if (
                obliquity(self.reference.affine).min() * 180 / pi
                > OBLIQUITY_THRESHOLD_DEG
            ):
                print('Reference affine axes are oblique.')
                M = self.reference.affine
                A = shape_zoom_affine(self.reference.shape,
                                      voxel_sizes(M), x_flip=False, y_flip=False)
                pre = M.dot(np.linalg.inv(A)).dot(LPS)

                if not moving:
                    moving = self.reference

            if (
                moving
                and obliquity(moving.affine).min() * 180 / pi
                > OBLIQUITY_THRESHOLD_DEG
            ):
                print('Moving affine axes are oblique.')
                M2 = moving.affine
                A2 = shape_zoom_affine(moving.shape,
                                       voxel_sizes(M2), x_flip=True, y_flip=True)
                post = A2.dot(np.linalg.inv(M2))

            # swapaxes is necessary, as axis 0 encodes series of transforms
            parameters = np.swapaxes(post.dot(self.matrix.copy().dot(pre)), 0, 1)
            parameters = parameters[:, :3, :].reshape((T.shape[0], -1))
            np.savetxt(filename, parameters, delimiter='\t', header="""\
3dvolreg matrices (DICOM-to-DICOM, row-by-row):""", fmt='%g')
            return filename

        # for FSL / FS information
        if not moving:
            moving = self.reference
        if isinstance(moving, str):
            moving = loadimg(moving)

        if fmt.lower() == 'fsl':
            # Adjust for reference image offset and orientation
            refswp, refspc = _fsl_aff_adapt(self.reference)
            pre = self.reference.affine.dot(
                np.linalg.inv(refspc).dot(np.linalg.inv(refswp)))

            # Adjust for moving image offset and orientation
            movswp, movspc = _fsl_aff_adapt(moving)
            post = np.linalg.inv(movswp).dot(movspc.dot(np.linalg.inv(
                moving.affine)))

            # Compose FSL transform
            mat = np.linalg.inv(
                np.swapaxes(post.dot(self.matrix.dot(pre)), 0, 1))

            if self.matrix.shape[0] > 1:
                for i in range(self.matrix.shape[0]):
                    np.savetxt('%s.%03d' % (filename, i), mat[i],
                               delimiter=' ', fmt='%g')
            else:
                np.savetxt(filename, mat[0], delimiter=' ', fmt='%g')
            return filename
        elif fmt.lower() == 'fs':
            # xform info
            lt = io.LinearTransform()
            lt['sigma'] = 1.
            lt['m_L'] = self.matrix
            lt['src'] = io.VolumeGeometry.from_image(moving)
            lt['dst'] = io.VolumeGeometry.from_image(self.reference)
            # to make LTA file format
            lta = io.LinearTransformArray()
            lta['type'] = 1  # RAS2RAS
            lta['xforms'].append(lt)

            with open(filename, 'w') as f:
                f.write(lta.to_string())
            return filename

        return super(Affine, self).to_filename(filename, fmt=fmt)


def load(filename, fmt='X5', reference=None):
    """Load a linear transform."""
    if fmt.lower() in ['itk', 'ants', 'elastix', 'nifty']:
        with open(filename) as itkfile:
            itkxfm = io.itk.ITKLinearTransformArray.from_fileobj(
                itkfile)

        matlist = []
        for xfm in itkxfm['xforms']:
            matrix = xfm['parameters']
            offset = xfm['offset']
            c_neg = from_matvec(np.eye(3), offset * -1.0)
            c_pos = from_matvec(np.eye(3), offset)
            matlist.append(LPS.dot(c_pos.dot(matrix.dot(c_neg.dot(LPS)))))
        matrix = np.stack(matlist)
    # elif fmt.lower() == 'afni':
    #     parameters = LPS.dot(self.matrix.dot(LPS))
    #     parameters = parameters[:3, :].reshape(-1).tolist()
    elif fmt.lower() == 'fs':
        with open(filename) as ltafile:
            lta = io.LinearTransformArray.from_fileobj(ltafile)
        if lta['nxforms'] > 1:
            raise NotImplementedError("Multiple transforms are not yet supported.")
        if lta['type'] != 1:
            warnings.warn("Converting LTA to RAS2RAS")
            lta.set_type(1)
        matrix = lta['xforms'][0]['m_L']
    elif fmt.lower() in ('x5', 'bids'):
        raise NotImplementedError
    else:
        raise NotImplementedError

    return Affine(matrix, reference=reference)


def _fsl_aff_adapt(space):
    """
    Adapt FSL affines.
    Calculates a matrix to convert from the original RAS image
    coordinates to FSL's internal coordinate system of transforms
    """
    aff = space.affine
    zooms = list(voxel_sizes(aff)) + [1]
    swp = np.eye(4)
    if np.linalg.det(aff) > 0:
        swp[0, 0] = -1.0
        swp[0, 3] = (space.shape[0] - 1) * zooms[0]
    return swp, np.diag(zooms)
