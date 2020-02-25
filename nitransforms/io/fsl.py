"""Read/write FSL's transforms."""
import os
import numpy as np
from pathlib import Path
from nibabel.affines import voxel_sizes

from .base import BaseLinearTransformList, LinearParameters, TransformFileError


class FSLLinearTransform(LinearParameters):
    """A string-based structure for FSL linear transforms."""

    def __str__(self):
        """Generate a string representation."""
        lines = [' '.join(['%g' % col for col in row])
                 for row in self.structarr['parameters']]
        return '\n'.join(lines + [''])

    def to_string(self):
        """Convert to a string directly writeable to file."""
        return self.__str__()

    @classmethod
    def from_ras(cls, ras, moving=None, reference=None):
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
        lines = [l.strip() for l in string.splitlines()
                 if l.strip()]
        if not lines or len(lines) < 4:
            raise TransformFileError

        print(lines)
        sa['parameters'] = np.genfromtxt(
            ['\n'.join(lines)], dtype=cls.dtype['parameters'])
        return tf


class FSLLinearTransformArray(BaseLinearTransformList):
    """A string-based structure for series of FSL linear transforms."""

    _inner_type = FSLLinearTransform

    def to_filename(self, filename):
        """Store this transform to a file with the appropriate format."""
        output_dir = Path(filename).parent
        output_dir.mkdir(exist_ok=True, parents=True)
        for i, xfm in enumerate(self.xforms):
            (output_dir / '.'.join((str(filename), '%03d' % i))).write_text(
                xfm.to_string())

    def to_ras(self, moving=None, reference=None):
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
        _self.xforms = [cls._inner_type.from_string(string)]
        return _self

    @classmethod
    def from_filename(cls, filename):
        """
        Read the struct from a file given its path.

        If the file does not exist, then indexed names
        with the zero-padded suffix ``.NNN`` are
        attempted, following FSL's MCFLIRT conventions.

        """
        if os.path.exists(str(filename)):
            return super().from_filename(filename)

        _xforms = []
        index = 0
        while os.path.exists('%s.%03d' % (filename, index)):
            with open('%s.%03d' % (filename, index)) as f:
                string = f.read()
            _xforms.append(cls._inner_type.from_string(string))
            index += 1

        if index == 0:
            raise FileNotFoundError(str(filename))
        _self = cls()
        _self.xforms = _xforms
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
