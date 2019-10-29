"""Read/write AFNI's transforms."""
from math import pi
import numpy as np
from nibabel.affines import obliquity, voxel_sizes

from ..patched import shape_zoom_affine
from .base import StringBasedStruct, LinearParameters

LPS = np.diag([-1, -1, 1, 1])
OBLIQUITY_THRESHOLD_DEG = 0.01


class AFNILinearTransform(LinearParameters):
    """A string-based structure for AFNI linear transforms."""

    def __str__(self):
        """Generate a string representation."""
        param = self.structarr['parameters']
        return '\t'.join(['%g' % p for p in param[:3, :].reshape(-1)])

    def to_string(self, banner=True):
        """Convert to a string directly writeable to file."""
        string = '%s\n' % self
        if banner:
            string = '\n'.join(("# 3dvolreg matrices (DICOM-to-DICOM, row-by-row):",
                                string))
        return string % self

    @classmethod
    def from_ras(cls, ras, moving, reference):
        """Create an ITK affine from a nitransform's RAS+ matrix."""
        ras = ras.copy()
        pre = LPS.copy()
        post = LPS.copy()
        if _is_oblique(reference.affine):
            print('Reference affine axes are oblique.')
            M = reference.affine
            A = shape_zoom_affine(reference.shape,
                                  voxel_sizes(M), x_flip=False, y_flip=False)
            pre = M.dot(np.linalg.inv(A)).dot(LPS)

        if _is_oblique(moving.affine):
            print('Moving affine axes are oblique.')
            M2 = moving.affine
            A2 = shape_zoom_affine(moving.shape,
                                   voxel_sizes(M2), x_flip=True, y_flip=True)
            post = A2.dot(np.linalg.inv(M2))

        # swapaxes is necessary, as axis 0 encodes series of transforms
        parameters = np.swapaxes(post.dot(ras.dot(pre)), 0, 1)

        tf = cls()
        tf.structarr['parameters'] = parameters.T
        return tf

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        tf = cls()
        sa = tf.structarr
        lines = [l for l in string.splitlines()
                 if l.strip()]

        if '3dvolreg matrices' in lines[0]:
            lines = lines[1:]  # Drop header

        parameters = np.eye(4, dtype='f4')
        parameters[:3, :] = np.genfromtxt(
            [lines[0].encode()], dtype=cls.dtype['parameters'])
        sa['parameters'] = parameters
        return tf


class AFNILinearTransformArray(StringBasedStruct):
    """A string-based structure for series of ITK linear transforms."""

    template_dtype = np.dtype([('nxforms', 'i4')])
    dtype = template_dtype
    _xforms = None

    def __init__(self,
                 xforms=None,
                 binaryblock=None,
                 endianness=None,
                 check=True):
        """Initialize with (optionally) a list of transforms."""
        super().__init__(binaryblock, endianness, check)
        self.xforms = [AFNILinearTransform(parameters=mat)
                       for mat in xforms or []]

    @property
    def xforms(self):
        """Get the list of internal ITKLinearTransforms."""
        return self._xforms

    @xforms.setter
    def xforms(self, value):
        self._xforms = list(value)

    def __getitem__(self, idx):
        """Allow dictionary access to the transforms."""
        if idx == 'xforms':
            return self._xforms
        if idx == 'nxforms':
            return len(self._xforms)
        raise KeyError(idx)

    def to_filename(self, filename):
        """Store this transform to a file with the appropriate format."""
        with open(str(filename), 'w') as f:
            f.write(self.to_string())

    def to_ras(self, moving, reference):
        """Return a nitransforms' internal RAS matrix."""
        return np.stack([xfm.to_ras(moving=moving, reference=reference)
                         for xfm in self.xforms])

    def to_string(self):
        """Convert to a string directly writeable to file."""
        strings = []
        for i, xfm in enumerate(self.xforms):
            strings.append(xfm.to_string(banner=(i == 0)))
        return '\n'.join(strings)

    @classmethod
    def from_fileobj(cls, fileobj, check=True):
        """Read the struct from a file object."""
        return cls.from_string(fileobj.read())

    @classmethod
    def from_ras(cls, ras, moving, reference):
        """Create an ITK affine from a nitransform's RAS+ matrix."""
        _self = cls()
        _self.xforms = [AFNILinearTransform.from_ras(
            ras[i, ...], moving=moving, reference=reference)
            for i in range(ras.shape[0])]
        return _self

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        _self = cls()
        _self.xforms = [AFNILinearTransform.from_string(l.strip())
                        for l in string.splitlines() if l.strip()]
        return _self


def _is_oblique(affine, thres=OBLIQUITY_THRESHOLD_DEG):
    return (obliquity(affine).min() * 180 / pi) > thres
