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
import numpy as np
import scipy.sparse as sparse
from nitransforms.base import (
    SurfaceMesh
)
import nibabel as nb
from scipy.spatial import KDTree
from nitransforms.base import TransformBase


class SurfaceTransformBase():
    """Generic surface transformation class"""
    __slots__ = ("_reference", "_moving")
    def __init__(self, reference, moving):
        """Instantiate a generic surface transform."""
        self._reference = reference
        self._moving = moving

    def __eq__(self, other):
        ref_coords_eq = (self.reference._coordinates == other.reference._coordinates).all()
        ref_tris_eq =  (self.reference._triangles == other.reference._triangles).all()
        mov_coords_eq = (self.moving._coordinates == other.moving._coordinates).all()
        mov_tris_eq = (self.moving._triangles == other.moving._triangles).all()
        return ref_coords_eq & ref_tris_eq & mov_coords_eq & mov_tris_eq

    def __invert__(self):
        return self.__class__(self.moving, self.reference)
    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, surface):
        self._reference = SurfaceMesh(surface)

    @property
    def moving(self):
        return self._moving

    @moving.setter
    def moving(self, surface):
        self._moving = SurfaceMesh(surface)
    @classmethod
    def from_filename(cls, reference_path, moving_path):
        """Create an Surface Index Transformation from a pair of surfaces with corresponding vertices."""
        reference = SurfaceMesh(nb.load(reference_path))
        moving = SurfaceMesh(nb.load(moving_path))
        return cls(reference, moving)

class SurfaceIndexTransform(SurfaceTransformBase):
    """Represents surface transformations in which the indices correspond and the coordinates differ."""

    __slots__ = ("_reference", "_moving")
    def __init__(self, reference, moving):
        """Instantiate a transform between two surfaces with corresponding vertices."""
        super().__init__(reference=reference, moving=moving)
        if (self._reference._triangles != self._moving._triangles).all():
            raise ValueError("Both surfaces for an index transform must have corresponding vertices.")

    def map(self, x, inverse=False):
        if inverse:
            source = self.reference
            dest = self.moving
        else:
            source = self.moving
            dest = self.reference

        s_tree = KDTree(source._coords)
        dists, matches = s_tree.query(x)
        if not np.allclose(dists, 0):
            raise NotImplementedError("Mapping on surfaces not implemented for coordinates that aren't vertices")
        return dest._coords[matches]

    def __add__(self, other):
        return self.__class__(self.reference, other.moving)

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, surface):
        self._reference = SurfaceMesh(surface)

    @property
    def moving(self):
        return self._moving

    @moving.setter
    def moving(self, surface):
        self._moving = SurfaceMesh(surface)

class SurfaceCoordinateTransform(SurfaceTransformBase):
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

    def apply(self, x, inverse=False, normalize="element"):
        """Apply the transform to surface data.

        Parameters
        ----------
        x : array-like, shape (..., nv1)
            Data to transform.
        inverse : bool, default=False
            Whether to apply the inverse transform. If True, ``x`` has shape
            (..., nv2), and the output will have shape (..., nv1).
        normalize : {"element", "sum", "none"}, default="element"
            Normalization strategy. If "element", the scale of each value in
            the output is comparable to each value of the input. If "sum", the
            sum of the output is comparable to the sum of the input. If
            "none", no normalization is applied.

        Returns
        -------
        y : array-like, shape (..., nv2)
            Transformed data.
        """
        if normalize not in ("element", "sum", "none"):
            raise ValueError("Invalid normalization strategy.")

        mat = self.mat.T if inverse else self.mat

        if normalize == "element":
            sum_ = mat.sum(axis=0)
            scale = np.zeros_like(sum_)
            mask = sum_ != 0
            scale[mask] = 1.0 / sum_[mask]
            mat = mat @ sparse.diags(scale)
        elif normalize == "sum":
            sum_ = mat.sum(axis=1)
            scale = np.zeros_like(sum_)
            mask = sum_ != 0
            scale[mask] = 1.0 / sum_[mask]
            mat = sparse.diags(scale) @ mat

        y = x @ mat
        return y

    def _to_hdf5(self, x5_root):
        """Write transform to HDF5 file."""
        xform = x5_root.create_group("Transform")
        xform.attrs["Type"] = "surface"
        xform.create_dataset("data", data=self.mat.data)
        xform.create_dataset("indices", data=self.mat.indices)
        xform.create_dataset("indptr", data=self.mat.indptr)
        xform.create_dataset("shape", data=self.mat.shape)

    def to_filename(self, filename, fmt=None):
        """Store the transform."""
        if fmt is None:
            fmt = "npz" if filename.endswith(".npz") else "X5"

        if fmt == "npz":
            sparse.save_npz(filename, self.mat)
            return filename

        with h5py.File(filename, "w") as out_file:
            out_file.attrs["Format"] = "X5"
            out_file.attrs["Version"] = np.uint16(1)
            root = out_file.create_group("/0")
            self._to_hdf5(root)

        return filename

    @classmethod
    def from_filename(cls, filename, fmt=None):
        """Load transform from file."""
        if fmt is None:
            fmt = "npz" if filename.endswith(".npz") else "X5"

        if fmt == "npz":
            return cls(sparse.load_npz(filename))

        if fmt != "X5":
            raise ValueError("Only npz and X5 formats are supported.")

        with h5py.File(filename, "r") as f:
            assert f.attrs["Format"] == "X5"
            xform = f["/0/Transform"]
            mat = sparse.csr_matrix(
                (xform["data"][()], xform["indices"][()], xform["indptr"][()]),
                shape=xform["shape"][()],
            )
        return cls(mat)
