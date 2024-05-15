# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Surface transforms."""

import h5py
import scipy.sparse as sparse

from nitransforms.base import TransformBase


class SurfaceTransform(TransformBase):
    """Represents transforms between surface spaces."""

    __slots__ = ("mat",)

    def __init__(self, mat):
        """Initialize the transform.

        Parameters
        ----------
        mat : array-like, shape (nv1, nv2)
            Sparse matrix representing the transform.
        """
        super().__init__()
        if isinstance(mat, sparse.csr_array):
            self.mat = mat
        else:
            self.mat = sparse.csr_array(mat)

    def apply(self, x, inverse=False):
        """Apply the transform to surface data.

        Parameters
        ----------
        x : array-like, shape (..., nv1)
            Data to transform.
        inverse : bool, default=False
            Whether to apply the inverse transform. If True, ``x`` has shape
            (..., nv2), and the output will have shape (..., nv1).

        Returns
        -------
        y : array-like, shape (..., nv2)
            Transformed data.
        """
        if inverse:
            return x @ self.mat.T
        return x @ self.mat

    def _to_hdf5(self, x5_root):
        """Write transform to HDF5 file."""
        xform = x5_root.create_group("Transform")
        xform.attrs["Type"] = "surface"
        xform.create_dataset("data", data=self.mat.data)
        xform.create_dataset("indices", data=self.mat.indices)
        xform.create_dataset("indptr", data=self.mat.indptr)
        xform.create_dataset("shape", data=self.mat.shape)

    @classmethod
    def from_filename(cls, filename, fmt="X5"):
        """Load transform from file."""
        if fmt != "X5":
            raise ValueError("Only X5 format is supported.")
        with h5py.File(filename, "r") as f:
            assert f.attrs["Format"] == "X5"
            xform = f["/0/Transform"]
            mat = sparse.csr_matrix(
                (xform["data"][()], xform["indices"][()], xform["indptr"][()]),
                shape=xform["shape"][()],
            )
            return cls(mat)
