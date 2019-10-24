"""Read/write linear transforms."""
import numpy as np
from nibabel.wrapstruct import LabeledWrapStruct as LWS


class LabeledWrapStruct(LWS):
    def __setitem__(self, item, value):
        self._structarr[item] = np.asanyarray(value)


class StringBasedStruct(LabeledWrapStruct):
    def __init__(self,
                 binaryblock=None,
                 endianness=None,
                 check=True):
        if binaryblock is not None and getattr(binaryblock, 'dtype',
                                               None) == self.dtype:
            self._structarr = binaryblock.copy()
            return
        super(StringBasedStruct, self).__init__(binaryblock, endianness, check)

    def __array__(self):
        return self._structarr
