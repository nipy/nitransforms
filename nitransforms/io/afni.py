"""Read/write AFNI's transforms."""
from math import pi
import numpy as np
from nibabel.affines import (
    from_matvec,
    obliquity,
    voxel_sizes,
)

from .base import (
    BaseLinearTransformList,
    DisplacementsField,
    LinearParameters,
    TransformFileError,
    _ensure_image,
)

LPS = np.diag([-1, -1, 1, 1])
OBLIQUITY_THRESHOLD_DEG = 0.01


class AFNILinearTransform(LinearParameters):
    """A string-based structure for AFNI linear transforms."""

    def __str__(self):
        """Generate a string representation."""
        param = self.structarr["parameters"]
        return "\t".join(["%g" % p for p in param[:3, :].reshape(-1)])

    def to_string(self, banner=True):
        """Convert to a string directly writeable to file."""
        string = "%s\n" % self
        if banner:
            string = "\n".join(
                ("# 3dvolreg matrices (DICOM-to-DICOM, row-by-row):", string)
            )
        return string % self

    @classmethod
    def from_ras(cls, ras, moving=None, reference=None):
        """Create an AFNI affine from a nitransform's RAS+ matrix."""
        pre = LPS
        post = LPS

        if reference is not None:
            reference = _ensure_image(reference)

        if reference is not None and _is_oblique(reference.affine):
            print("Reference affine axes are oblique.")
            ras = ras @ _afni_warpdrive(reference.affine, ras=True, forward=False)

        if moving is not None:
            moving = _ensure_image(moving)

        if moving is not None and _is_oblique(moving.affine):
            print("Moving affine axes are oblique.")
            ras = _afni_warpdrive(reference.affine, ras=True) @ ras

        # swapaxes is necessary, as axis 0 encodes series of transforms
        parameters = np.swapaxes(post @ ras @ pre, 0, 1)

        tf = cls()
        tf.structarr["parameters"] = parameters.T
        return tf

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        tf = cls()
        sa = tf.structarr
        lines = [
            line
            for line in string.splitlines()
            if line.strip()
            and not (line.startswith("#") or "3dvolreg matrices" in line)
        ]

        if not lines:
            raise TransformFileError

        parameters = np.vstack(
            (
                np.genfromtxt([lines[0].encode()], dtype="f8").reshape((3, 4)),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        sa["parameters"] = parameters
        return tf

    def to_ras(self, moving=None, reference=None):
        """Return a nitransforms internal RAS+ matrix."""
        pre = LPS
        post = LPS

        if reference is not None:
            reference = _ensure_image(reference)

        if reference is not None and _is_oblique(reference.affine):
            raise NotImplementedError

        if moving is not None:
            moving = _ensure_image(moving)

        if moving is not None and _is_oblique(moving.affine):
            raise NotImplementedError

        # swapaxes is necessary, as axis 0 encodes series of transforms
        return post @ np.swapaxes(self.structarr["parameters"].T, 0, 1) @ pre


class AFNILinearTransformArray(BaseLinearTransformList):
    """A string-based structure for series of AFNI linear transforms."""

    _inner_type = AFNILinearTransform

    def to_ras(self, moving=None, reference=None):
        """Return a nitransforms' internal RAS matrix."""
        return np.stack(
            [xfm.to_ras(moving=moving, reference=reference) for xfm in self.xforms]
        )

    def to_string(self):
        """Convert to a string directly writeable to file."""
        strings = []
        for i, xfm in enumerate(self.xforms):
            lines = [
                line.strip()
                for line in xfm.to_string(banner=(i == 0)).splitlines()
                if line.strip()
            ]
            strings += lines
        return "\n".join(strings)

    @classmethod
    def from_ras(cls, ras, moving=None, reference=None):
        """Create an ITK affine from a nitransform's RAS+ matrix."""
        _self = cls()
        _self.xforms = [
            cls._inner_type.from_ras(ras[i, ...], moving=moving, reference=reference)
            for i in range(ras.shape[0])
        ]
        return _self

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        _self = cls()

        lines = [
            line.strip()
            for line in string.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if not lines:
            raise TransformFileError("Input string is empty.")

        _self.xforms = [cls._inner_type.from_string(line) for line in lines]
        return _self


class AFNIDisplacementsField(DisplacementsField):
    """A data structure representing displacements fields."""

    @classmethod
    def from_image(cls, imgobj):
        """Import a displacements field from a NIfTI file."""
        hdr = imgobj.header.copy()
        shape = hdr.get_data_shape()

        if len(shape) != 5 or shape[-2] != 1 or not shape[-1] in (2, 3):
            raise TransformFileError(
                'Displacements field "%s" does not come from AFNI.'
                % imgobj.file_map["image"].filename
            )

        field = np.squeeze(np.asanyarray(imgobj.dataobj))
        field[..., (0, 1)] *= -1.0

        return imgobj.__class__(field, imgobj.affine, hdr)


def _is_oblique(affine, thres=OBLIQUITY_THRESHOLD_DEG):
    return (obliquity(affine).min() * 180 / pi) > thres


def _afni_warpdrive(nii, forward=True, ras=False):
    """
    Calculate AFNI's ``WARPDRIVE_MATVEC_FOR_000000`` (de)obliquing affine.

    Parameters
    ----------
    oblique : 4x4 numpy.array
        affine that is not aligned to the cardinal axes.
    plumb : 4x4 numpy.array
        corresponding affine that is aligned to the cardinal axes.
    forward : :obj:`bool`
        Transforms the affine of oblique into an AFNI's plumb (if ``True``)
        or viceversa plumb -> oblique (if ``false``).

    Returns
    -------
    plumb_to_oblique : 4x4 numpy.array
        the matrix that pre-pended to the plumb affine rotates it
        to be oblique.

    """
    oblique = nii.affine
    plumb = oblique[:3, :3] / np.abs(oblique[:3, :3]).max(0)
    plumb[np.abs(plumb) < 1.0] = 0
    plumb *= voxel_sizes(oblique)

    R = from_matvec(plumb @ np.linalg.inv(oblique[:3, :3]), (0, 0, 0))
    plumb_orig = np.linalg.inv(R[:3, :3]) @ oblique[:3, 3]
    print(plumb_orig)
    R[:3, 3] = R[:3, :3] @ (plumb_orig - oblique[:3, 3])
    if not ras:
        # Change sign to match AFNI's warpdrive_matvec signs
        B = np.ones((2, 2))
        R *= np.block([[B, -1.0 * B], [-1.0 * B, B]])

    return R if forward else np.linalg.inv(R)


def _afni_header(nii, field="WARPDRIVE_MATVEC_FOR_000000"):
    from lxml import etree
    root = etree.fromstring(nii.header.extensions[0].get_content().decode())
    retval = np.fromstring(
        root.find(f".//*[@atr_name='{field}']").text,
        sep="\n",
        dtype="float32"
    ).reshape((3, 4))
    return np.vstack((retval, (0, 0, 0, 1)))
