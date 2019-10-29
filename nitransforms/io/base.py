"""Read/write linear transforms."""
import numpy as np
from scipy.io.matlab.miobase import get_matfile_version
from scipy.io.matlab.mio4 import MatFile4Reader
from scipy.io.matlab.mio5 import MatFile5Reader

from ..patched import LabeledWrapStruct


class TransformFileError(Exception):
    """A custom exception for transform files."""


class StringBasedStruct(LabeledWrapStruct):
    """File data structure from text files."""

    def __init__(self,
                 binaryblock=None,
                 endianness=None,
                 check=True):
        """Create a data structure based off of a string."""
        _dtype = getattr(binaryblock, 'dtype', None)
        if binaryblock is not None and _dtype == self.dtype:
            self._structarr = binaryblock.copy()
            return
        super(StringBasedStruct, self).__init__(binaryblock, endianness, check)

    def __array__(self):
        """Return the internal structure array."""
        return self._structarr


class LinearParameters(StringBasedStruct):
    """
    A string-based structure for linear transforms.

    Examples
    --------
    >>> lp = LinearParameters()
    >>> np.all(lp.structarr['parameters'] == np.eye(4))
    True

    >>> p = np.diag([2., 2., 2., 1.])
    >>> lp = LinearParameters(p)
    >>> np.all(lp.structarr['parameters'] == p)
    True

    """

    template_dtype = np.dtype([
        ('parameters', 'f8', (4, 4)),
    ])
    dtype = template_dtype

    def __init__(self, parameters=None):
        """Initialize with default parameters."""
        super().__init__()
        self.structarr['parameters'] = np.eye(4)
        if parameters is not None:
            self.structarr['parameters'] = parameters

    def to_filename(self, filename):
        """Store this transform to a file with the appropriate format."""
        with open(str(filename), 'w') as f:
            f.write(self.to_string())

    def to_ras(self):
        """Return a nitransforms internal RAS+ matrix."""
        raise NotImplementedError

    @classmethod
    def from_fileobj(cls, fileobj, check=True):
        """Read the struct from a file object."""
        return cls.from_string(fileobj.read())

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        raise NotImplementedError


def _read_mat(byte_stream):
    mjv, _ = get_matfile_version(byte_stream)
    if mjv == 0:
        reader = MatFile4Reader(byte_stream)
    elif mjv == 1:
        reader = MatFile5Reader(byte_stream)
    elif mjv == 2:
        raise TransformFileError('Please use HDF reader for Matlab v7.3 files')
    else:
        raise TransformFileError('Not a Matlab file.')
    return reader.get_variables()
