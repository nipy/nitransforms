"""Read/write linear transforms."""
from pathlib import Path
import numpy as np
from nibabel import load as loadimg

from ..patched import LabeledWrapStruct


class TransformIOError(IOError):
    """General I/O exception while reading/writing transforms."""


class TransformFileError(TransformIOError):
    """Specific I/O exception when a file does not meet the expected format."""


class StringBasedStruct(LabeledWrapStruct):
    """File data structure from text files."""

    def __init__(self, binaryblock=None, endianness=None, check=True):
        """Create a data structure based off of a string."""
        _dtype = getattr(binaryblock, "dtype", None)
        if binaryblock is not None and _dtype == self.dtype:
            self._structarr = binaryblock.copy()
            return
        super().__init__(binaryblock, endianness, check)

    def __array__(self):
        """Return the internal structure array."""
        return self._structarr

    def to_string(self):
        """Convert to a string directly writeable to file."""
        raise NotImplementedError

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        raise NotImplementedError


class LinearTransformStruct(StringBasedStruct):
    """File data structure from linear transforms."""

    def to_filename(self, filename):
        """Store this transform to a file with the appropriate format."""
        with open(str(filename), "w") as f:
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
    def from_ras(cls, ras, moving=None, reference=None):
        """Create an affine from a nitransform's RAS+ matrix."""
        raise NotImplementedError


class LinearParameters(LinearTransformStruct):
    """
    A string-based structure for linear transforms.

    Examples
    --------
    >>> lp = LinearParameters()
    >>> np.array_equal(lp.structarr['parameters'], np.eye(4))
    True

    >>> p = np.diag([2., 2., 2., 1.])
    >>> lp = LinearParameters(p)
    >>> np.array_equal(lp.structarr['parameters'], p)
    True

    """

    template_dtype = np.dtype([("parameters", "f8", (4, 4))])
    dtype = template_dtype

    def __init__(self, parameters=None):
        """
        Initialize with default parameters.


        """
        super().__init__()
        self.structarr["parameters"] = np.eye(4)
        if parameters is not None:
            self.structarr["parameters"] = parameters


class BaseLinearTransformList(LinearTransformStruct):
    """A string-based structure for series of linear transforms."""

    template_dtype = np.dtype([("nxforms", "i4")])
    dtype = template_dtype
    _xforms = None
    _inner_type = LinearParameters

    def __init__(self, xforms=None, binaryblock=None, endianness=None, check=True):
        """Initialize with (optionally) a list of transforms."""
        super().__init__(binaryblock, endianness, check)
        self.xforms = [self._inner_type(parameters=mat) for mat in xforms or []]

    @property
    def xforms(self):
        """Get the list of internal transforms."""
        return self._xforms

    @xforms.setter
    def xforms(self, value):
        self._xforms = list(value)

    def __getitem__(self, idx):
        """Allow dictionary access to the transforms."""
        if idx == "xforms":
            return self._xforms
        if idx == "nxforms":
            return len(self._xforms)
        raise KeyError(idx)


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

    @classmethod
    def to_filename(cls, img, filename):
        """Export a displacements field to a NIfTI file."""
        imgobj = cls.to_image(img)
        imgobj.to_filename(filename)

    @classmethod
    def to_image(cls, imgobj):
        """Export a displacements field image from a nitransforms image object."""
        raise NotImplementedError


def _ensure_image(img):
    if isinstance(img, (str, Path)):
        return loadimg(img)
    return img
