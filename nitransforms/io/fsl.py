"""Read/write FSL's transforms."""
from io import StringIO
import numpy as np
from nibabel.affines import voxel_sizes

from .base import LinearParameters, StringBasedStruct


class FSLLinearTransform(LinearParameters):
    """A string-based structure for FSL linear transforms."""

    def __str__(self):
        """Generate a string representation."""
        param = self.structarr['parameters']
        return '\t'.join(['%g' % p for p in param[:3, :].reshape(-1)])

    def to_string(self):
        """Convert to a string directly writeable to file."""
        with StringIO() as f:
            np.savetxt(f, self.structarr['parameters'],
                       delimiter=' ', fmt='%g')
            string = f.getvalue()
        return string

    @classmethod
    def from_ras(cls, ras, moving, reference):
        """Create an ITK affine from a nitransform's RAS+ matrix."""
        # Adjust for reference image offset and orientation
        refswp, refspc = _fsl_aff_adapt(reference)
        pre = reference.affine.dot(
            np.linalg.inv(refspc).dot(np.linalg.inv(refswp)))

        # Adjust for moving image offset and orientation
        movswp, movspc = _fsl_aff_adapt(moving)
        post = np.linalg.inv(movswp).dot(movspc.dot(np.linalg.inv(
            moving.affine)))

        # Compose FSL transform
        mat = np.linalg.inv(
            np.swapaxes(post.dot(ras.dot(pre)), 0, 1))

        tf = cls()
        tf.structarr['parameters'] = mat.T
        return tf

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        tf = cls()
        sa = tf.structarr
        lines = [l.encode() for l in string.splitlines()
                 if l.strip()]

        if '3dvolreg matrices' in lines[0]:
            lines = lines[1:]  # Drop header

        parameters = np.eye(4, dtype='f4')
        parameters = np.genfromtxt(
            lines, dtype=cls.dtype['parameters'])
        sa['parameters'] = parameters
        return tf


class FSLLinearTransformArray(StringBasedStruct):
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
        self.xforms = [FSLLinearTransform(parameters=mat)
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
        if len(self.xforms) == 1:
            self.xforms[0].to_filename(filename)
            return

        for i, xfm in enumerate(self.xforms):
            with open('%s.%03d' % (filename, i), 'w') as f:
                f.write(xfm.to_string())

    def to_ras(self, moving, reference):
        """Return a nitransforms' internal RAS matrix."""
        return np.stack([xfm.to_ras(moving=moving, reference=reference)
                         for xfm in self.xforms])

    def to_string(self):
        """Convert to a string directly writeable to file."""
        return '\n\n'.join([xfm.to_string() for xfm in self.xforms])

    @classmethod
    def from_fileobj(cls, fileobj, check=True):
        """Read the struct from a file object."""
        return cls.from_string(fileobj.read())

    @classmethod
    def from_ras(cls, ras, moving, reference):
        """Create an ITK affine from a nitransform's RAS+ matrix."""
        _self = cls()
        _self.xforms = [FSLLinearTransform.from_ras(
            ras[i, ...], moving=moving, reference=reference)
            for i in range(ras.shape[0])]
        return _self

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        _self = cls()
        _self.xforms = [FSLLinearTransform.from_string(l.strip())
                        for l in string.splitlines() if l.strip()]
        return _self


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
