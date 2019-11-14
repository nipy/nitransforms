"""Read/write linear transforms."""
import numpy as np
from nibabel import load as loadimg
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

    def to_ras(self, moving=None, reference=None):
        """Return a nitransforms internal RAS+ matrix."""
        raise NotImplementedError

    @classmethod
    def from_filename(cls, filename):
        """Read the struct from a file given its path."""
        with open(str(filename)) as f:
            string = f.read()
        return cls.from_string(string)

    @classmethod
    def from_fileobj(cls, fileobj, check=True):
        """Read the struct from a file object."""
        return cls.from_string(fileobj.read())

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        raise NotImplementedError


class BaseLinearTransformList(StringBasedStruct):
    """A string-based structure for series of linear transforms."""

    template_dtype = np.dtype([('nxforms', 'i4')])
    dtype = template_dtype
    _xforms = None
    _inner_type = LinearParameters

    def __init__(self,
                 xforms=None,
                 binaryblock=None,
                 endianness=None,
                 check=True):
        """Initialize with (optionally) a list of transforms."""
        super().__init__(binaryblock, endianness, check)
        self.xforms = [self._inner_type(parameters=mat)
                       for mat in xforms or []]

    @property
    def xforms(self):
        """Get the list of internal transforms."""
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

    def to_ras(self, moving=None, reference=None):
        """Return a nitransforms' internal RAS matrix."""
        raise NotImplementedError

    def to_string(self):
        """Convert to a string directly writeable to file."""
        raise NotImplementedError

    @classmethod
    def from_filename(cls, filename):
        """Read the struct from a file given its path."""
        with open(str(filename)) as f:
            string = f.read()
        return cls.from_string(string)

    @classmethod
    def from_fileobj(cls, fileobj, check=True):
        """Read the struct from a file object."""
        return cls.from_string(fileobj.read())

    @classmethod
    def from_ras(cls, ras, moving=None, reference=None):
        """Create an ITK affine from a nitransform's RAS+ matrix."""
        raise NotImplementedError

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        raise NotImplementedError


class DisplacementsField:
    """A data structure representing displacements fields."""

    @classmethod
    def from_filename(cls, filename):
        """Import a displacements field from a NIfTI file."""
        imgobj = loadimg(str(filename))
        return cls.from_image(imgobj)

    @classmethod
    def from_image(cls, imgobj):
        """Import a displacements field from a nibabel image object."""
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
