"""Read/write linear transforms."""
from scipy.io.matlab.miobase import get_matfile_version
from scipy.io.matlab.mio4 import MatFile4Reader  # , MatFile4Writer
from scipy.io.matlab.mio5 import MatFile5Reader  # , MatFile5Writer

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


def _read_mat(byte_stream):
    mjv, _ = get_matfile_version(byte_stream)
    if mjv == 0:
        reader = MatFile4Reader(byte_stream)
    elif mjv == 1:
        reader = MatFile5Reader(byte_stream)
    elif mjv == 2:
        raise TransformFileError('Please use HDF reader for matlab v7.3 files')
    return reader.get_variables()
