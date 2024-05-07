"""Read/write x5 transforms."""
import warnings
import numpy as np
from scipy.io import loadmat as _read_mat, savemat as _save_mat
from h5py import File as H5File
from nibabel import Nifti1Header, Nifti1Image
from nibabel.affines import from_matvec
from nitransforms.io.base import (
    BaseLinearTransformList,
    DisplacementsField,
    LinearParameters,
    TransformIOError,
    TransformFileError,
)

LPS = np.diag([-1, -1, 1, 1])

class X5LinearTransform(LinearParameters):
    """A string-based structure for X5 linear transforms."""

    template_dtype = np.dtype(
        [
            ("type", "i4"),
            ("index", "i4"),
            ("parameters", "f8", (4, 4)),
            ("offset", "f4", 3),  # Center of rotation
        ]
    )
    dtype = template_dtype

    def __init__(self, parameters=None, offset=None):
        return

    def __str__(self):
        return

    def to_filename(self, filename):
        """Store this transform to a file with the appropriate format."""
        sa = self.structarr
        affine = np.array(
                np.hstack(
                    (sa["parameters"][:3, :3].reshape(-1), sa["parameters"][:3, 3])
                )[..., np.newaxis],
                dtype="f8",
            )
        return
    
    @classmethod
    def from_filename(cls, filename):
        """Read the struct from a X5 file given its path."""
        if str(filename).endswith(".h5"):
            with H5File(str(filename)) as f:
                return cls.from_h5obj(f)
