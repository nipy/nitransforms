"""Read/write AFNI's transforms."""
from math import pi
import warnings
import numpy as np
from nibabel.affines import obliquity, voxel_sizes, from_matvec

from ..patched import shape_zoom_affine
from .base import (
    BaseLinearTransformList,
    DisplacementsField,
    LinearParameters,
    TransformFileError,
)

LPS = np.diag([-1, -1, 1, 1])
OBLIQUITY_THRESHOLD_DEG = 0.01
B = np.ones((2, 2))
AFNI_SIGNS = np.block([[B, -1.0 * B], [-1.0 * B, B]])


class AFNILinearTransform(LinearParameters):
    """A string-based structure for AFNI linear transforms."""

    def __str__(self):
        """Generate a string representation."""
        param = self.structarr['parameters']
        return '\t'.join(['%g' % p for p in param[:3, :].reshape(-1)])

    def to_ras(self, moving=None, reference=None):
        """Convert to RAS+ coordinate system."""
        afni = self.structarr['parameters'].copy()
        # swapaxes is necessary, as axis 0 encodes series of transforms
        return _afni2ras(afni, moving=moving, reference=reference)

    def to_string(self, banner=True):
        """Convert to a string directly writeable to file."""
        string = '%s\n' % self
        if banner:
            string = '\n'.join(("# 3dvolreg matrices (DICOM-to-DICOM, row-by-row):",
                                string))
        return string % self

    @classmethod
    def from_ras(cls, ras, moving=None, reference=None):
        """Create an AFNI affine from a nitransform's RAS+ matrix."""
        tf = cls()
        tf.structarr['parameters'] = _ras2afni(ras, moving, reference)
        return tf

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        tf = cls()
        sa = tf.structarr
        lines = [
            l for l in string.splitlines()
            if l.strip() and not (l.startswith('#') or '3dvolreg matrices' in l)
        ]

        if not lines:
            raise TransformFileError

        parameters = np.vstack((
            np.genfromtxt([lines[0].encode()],
                          dtype='f8').reshape((3, 4)),
            (0., 0., 0., 1.)))
        sa['parameters'] = parameters
        return tf


class AFNILinearTransformArray(BaseLinearTransformList):
    """A string-based structure for series of AFNI linear transforms."""

    _inner_type = AFNILinearTransform

    def to_ras(self, moving=None, reference=None):
        """Return a nitransforms' internal RAS matrix."""
        return np.stack([xfm.to_ras(moving=moving, reference=reference)
                         for xfm in self.xforms])

    def to_string(self):
        """Convert to a string directly writeable to file."""
        strings = []
        for i, xfm in enumerate(self.xforms):
            lines = [
                l.strip()
                for l in xfm.to_string(banner=(i == 0)).splitlines()
                if l.strip()]
            strings += lines + ['']
        return '\n'.join(strings)

    @classmethod
    def from_ras(cls, ras, moving=None, reference=None):
        """Create an ITK affine from a nitransform's RAS+ matrix."""
        _self = cls()
        _self.xforms = [cls._inner_type.from_ras(
            ras[i, ...], moving=moving, reference=reference)
            for i in range(ras.shape[0])]
        return _self

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        _self = cls()

        lines = [l.strip() for l in string.splitlines()
                 if l.strip() and not l.startswith('#')]
        if not lines:
            raise TransformFileError('Input string is empty.')

        _self.xforms = [cls._inner_type.from_string(l)
                        for l in lines]
        return _self


class AFNIDisplacementsField(DisplacementsField):
    """A data structure representing displacements fields."""

    @classmethod
    def from_image(cls, imgobj):
        """Import a displacements field from a NIfTI file."""
        hdr = imgobj.header.copy()
        shape = hdr.get_data_shape()

        if (
            len(shape) != 5 or
            shape[-2] != 1 or
            not shape[-1] in (2, 3)
        ):
            raise TransformFileError(
                'Displacements field "%s" does not come from AFNI.' %
                imgobj.file_map['image'].filename)

        field = np.squeeze(np.asanyarray(imgobj.dataobj))
        field[..., (0, 1)] *= -1.0

        return imgobj.__class__(field, imgobj.affine, hdr)


def _is_oblique(affine, thres=OBLIQUITY_THRESHOLD_DEG):
    return (obliquity(affine).min() * 180 / pi) > thres


def _afni_warpdrive_for(oblique, plumb, offset=True, inv=False):
    """
    Calculate AFNI's ``WARPDRIVE_MATVEC_FOR_000000`` (de)obliquing affine.

    Parameters
    ----------
    oblique : 4x4 numpy.array
        affine that is not aligned to the cardinal axes.
    plumb : 4x4 numpy.array
        corresponding affine that is aligned to the cardinal axes.


    Returns
    -------
    plumb_to_oblique : 4x4 numpy.array
        the matrix that pre-pended to the plumb affine rotates it
        to be oblique.

    """
    R = np.linalg.inv(plumb[:3, :3]).dot(oblique[:3, :3])
    origin = oblique[:3, 3] - R.dot(oblique[:3, 3])
    if offset is False:
        origin = np.zeros(3)
    matvec_inv = from_matvec(R, origin)  # * AFNI_SIGNS
    if not inv:
        return np.linalg.inv(matvec_inv)
    return matvec_inv


def _ras2afni(ras, moving=None, reference=None):
    """
    Convert from RAS+ to AFNI matrix.

    inverse : bool
        if ``False`` (default), return the matrix to rotate plumb
        onto oblique (AFNI's ``WARPDRIVE_MATVEC_INV_000000``);
        if ``True``, return the matrix to rotate oblique onto
        plumb (AFNI's ``WARPDRIVE_MATVEC_FOR_000000``).

    """
    ras = ras.copy()
    pre = np.eye(4)
    post = np.eye(4)
    if reference is not None and _is_oblique(reference.affine):
        warnings.warn('Reference affine axes are oblique.')
        M = reference.affine
        plumb = shape_zoom_affine(reference.shape, voxel_sizes(M))
        # Prepend the MATVEC_INV - AFNI will append MATVEC_FOR
        pre = _afni_warpdrive_for(M, plumb, offset=False, inv=True)

    if moving is not None and _is_oblique(moving.affine):
        warnings.warn('Moving affine axes are oblique.')
        M = moving.affine
        plumb = shape_zoom_affine(moving.shape, voxel_sizes(M))
        # Append the MATVEC_FOR - AFNI will append MATVEC_INV
        post = _afni_warpdrive_for(M, plumb, offset=False)

    afni_ras = np.swapaxes(post.dot(ras.dot(pre)), 0, 1).T

    # Combine oblique/deoblique matrices into RAS+ matrix
    return np.swapaxes(LPS.dot(afni_ras.dot(LPS)), 0, 1).T


def _afni2ras(afni, moving=None, reference=None):
    """
    Convert from RAS+ to AFNI matrix.

    inverse : bool
        if ``False`` (default), return the matrix to rotate plumb
        onto oblique (AFNI's ``WARPDRIVE_MATVEC_INV_000000``);
        if ``True``, return the matrix to rotate oblique onto
        plumb (AFNI's ``WARPDRIVE_MATVEC_FOR_000000``).

    """
    afni = afni.copy()
    pre = np.eye(4)
    post = np.eye(4)
    if reference is not None and _is_oblique(reference.affine):
        warnings.warn('Reference affine axes are oblique.')
        M = reference.affine
        plumb = shape_zoom_affine(reference.shape, voxel_sizes(M))
        # Append the MATVEC_FOR - AFNI would add it implicitly
        pre = _afni_warpdrive_for(M, plumb, offset=False)

    if moving is not None and _is_oblique(moving.affine):
        warnings.warn('Moving affine axes are oblique.')
        M = moving.affine
        plumb = shape_zoom_affine(moving.shape, voxel_sizes(M))
        # Prepend the MATVEC_INV - AFNI will add it implicitly
        post = _afni_warpdrive_for(M, plumb, offset=False, inv=False)

    afni_ras = np.swapaxes(post.dot(afni.dot(pre)), 0, 1).T

    # Combine oblique/deoblique matrices into RAS+ matrix
    return np.swapaxes(LPS.dot(afni_ras.dot(LPS)), 0, 1).T
