# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Surface transforms."""

import warnings
import h5py
import numpy as np
import scipy.sparse as sparse
from nitransforms.base import (
    SurfaceMesh
)
import nibabel as nb
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist


class SurfaceTransformBase():
    """Generic surface transformation class"""
    __slots__ = ("_reference", "_moving")

    def __init__(self, reference, moving):
        """Instantiate a generic surface transform."""
        self._reference = reference
        self._moving = moving

    def __eq__(self, other):
        ref_coords_eq = (self.reference._coords == other.reference._coords).all()
        ref_tris_eq = (self.reference._triangles == other.reference._triangles).all()
        mov_coords_eq = (self.moving._coords == other.moving._coords).all()
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
        """Create an Surface Index Transformation from a pair of surfaces with corresponding
         vertices."""
        reference = SurfaceMesh(nb.load(reference_path))
        moving = SurfaceMesh(nb.load(moving_path))
        return cls(reference, moving)


class SurfaceCoordinateTransform(SurfaceTransformBase):
    """Represents surface transformations in which the indices correspond and the coordinates
     differ."""

    __slots__ = ("_reference", "_moving")

    def __init__(self, reference, moving):
        """Instantiate a transform between two surfaces with corresponding vertices."""
        super().__init__(reference=reference, moving=moving)
        if (self._reference._triangles != self._moving._triangles).all():
            raise ValueError("Both surfaces for an index transform must have corresponding"
                             " vertices.")

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
            raise NotImplementedError("Mapping on surfaces not implemented for coordinates that"
                                      " aren't vertices")
        return dest._coords[matches]

    def __add__(self, other):
        if isinstance(other, SurfaceCoordinateTransform):
            return self.__class__(self.reference, other.moving)
        else:
            raise NotImplementedError

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


class SurfaceResampler(SurfaceTransformBase):
    """Represents transformations in which the coordinate space remains the same and the indicies
     change."""

    __slots__ = ("mat", 'interpolation_method')

    def __init__(self, reference, moving, interpolation_method='barycentric', mat=None):
        """Initialize the transform.

        Parameters
        ----------
        reference: spherical surface of the reference space.
            Output will have number of indices equal to the number of indicies in this surface.
            Both reference and moving should be in the same coordinate space.
        moving: spherical surface that will be resampled.
            Both reference and moving should be in the same coordinate space.
        mat : array-like, shape (nv1, nv2)
            Sparse matrix representing the transform.
        interpolation_method : str
            Only barycentric is currently implemented
        """
        super().__init__(reference, moving)
        self.interpolation_method = interpolation_method

        # TODO: should we deal with the case where reference and moving are the same?

        # we're calculating the interpolation in the init so that we can ensure
        # that it only has to be calculated once and will always be saved with the
        # transform
        if mat is None:
            m_tree = KDTree(self.moving._coords)
            kmr_dists, kmr_closest = m_tree.query(self.reference._coords, k=10)

            # invert the triangles to generate a lookup table from vertices to triangle index
            tri_lut = {}
            for i, idxs in enumerate(self.moving._triangles):
                for x in idxs:
                    if x not in tri_lut:
                        tri_lut[x] = [i]
                    else:
                        tri_lut[x].append(i)

            # calculate the barycentric interpolation weights
            bc_weights = []
            enclosing = []
            for _, (point, kmrv) in enumerate(zip(self.reference._coords, kmr_closest)):
                close_tris = _find_close_tris(kmrv, tri_lut, self.moving)
                ww, ee = _find_weights(point, close_tris, m_tree)
                bc_weights.append(ww)
                enclosing.append(ee)

            # build sparse matrix
            # commenting out code for barycentric nearest neighbor
            # bary_nearest = []
            mat = sparse.lil_array((self.reference._npoints, self.moving._npoints))
            for s_ix, dd in enumerate(bc_weights):
                for k, v in dd.items():
                    mat[s_ix, k] = v
                # bary_nearest.append(
                #   np.array(list(dd.keys()))[np.array(list(dd.values())).argmax()]
                # )
            # bary_nearest = np.array(bary_nearest)
            # transpose so that number of out vertices is columns
            self.mat = sparse.csr_array(mat.T)
        else:
            if isinstance(mat, sparse.csr_array):
                self.mat = mat
            else:
                self.mat = sparse.csr_array(mat)

    def map(self, x, inverse=False):
        return x

    def __add__(self, other):

        if (isinstance(other, SurfaceResampler)
                and (other.interpolation_method == self.interpolation_method)):
            return self.__class__(
                self.reference,
                other.moving,
                interpolation_method=self.interpolation_method
            )
        else:
            raise NotImplementedError

    def __invert__(self):
        return self.__class__(
            self.moving,
            self.reference,
            interpolation_method=self.interpolation_method
        )

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
        xform.attrs["Type"] = "SurfaceResampling"
        xform.attrs['interpolation_method'] = self.interpolation_method
        xform.create_dataset("mat_data", data=self.mat.data)
        xform.create_dataset("mat_indices", data=self.mat.indices)
        xform.create_dataset("mat_indptr", data=self.mat.indptr)
        xform.create_dataset("mat_shape", data=self.mat.shape)
        xform.create_dataset("reference_coordinates", data=self.reference._coords)
        xform.create_dataset("reference_triangles", data=self.reference._triangles)
        xform.create_dataset("moving_coordinates", data=self.moving._coords)
        xform.create_dataset("moving_triangles", data=self.moving._triangles)

    def to_filename(self, filename, fmt=None):
        """Store the transform."""
        if fmt is None:
            fmt = "npz" if filename.endswith(".npz") else "X5"

        if fmt == "npz":
            raise NotImplementedError
            # sparse.save_npz(filename, self.mat)
            # return filename

        with h5py.File(filename, "w") as out_file:
            out_file.attrs["Format"] = "X5"
            out_file.attrs["Version"] = np.uint16(1)
            root = out_file.create_group("/0")
            self._to_hdf5(root)

        return filename

    @classmethod
    def from_filename(cls, filename=None, reference_file=None, moving_file=None,
                      fmt=None, interpolation_method=None):
        """Load transform from file."""
        if filename is None:
            if reference_file is None or moving_file is None:
                raise ValueError("You must pass either a X5 file or a pair of reference and moving"
                                 " surfaces.")
            else:
                if interpolation_method is None:
                    interpolation_method = 'barycentric'
                return cls(SurfaceMesh(nb.load(reference_file)),
                           SurfaceMesh(nb.load(moving_file)),
                           interpolation_method=interpolation_method)
        else:
            if fmt is None:
                fmt = "npz" if filename.endswith(".npz") else "X5"

            if fmt == "npz":
                raise NotImplementedError
                # return cls(sparse.load_npz(filename))

            if fmt != "X5":
                raise ValueError("Only npz and X5 formats are supported.")

            with h5py.File(filename, "r") as f:
                assert f.attrs["Format"] == "X5"
                xform = f["/0/Transform"]
                mat = sparse.csr_matrix(
                    (xform["mat_data"][()], xform["mat_indices"][()], xform["mat_indptr"][()]),
                    shape=xform["mat_shape"][()],
                )
                reference = SurfaceMesh.from_arrays(
                    xform['reference_coordinates'],
                    xform['reference_triangles']
                )
                moving = SurfaceMesh.from_arrays(
                    xform['moving_coordinates'],
                    xform['moving_triangles']
                )
                interpolation_method = xform.attrs['interpolation_method']
            return cls(reference, moving, interpolation_method=interpolation_method, mat=mat)


def _pointsToTriangles(points, triangles):
    """Implementation that vectorizes project of a point to a set of triangles.
    from: https://stackoverflow.com/a/32529589
    """
    with np.errstate(all='ignore'):
        # Unpack triangle points
        p0, p1, p2 = np.asarray(triangles).swapaxes(0, 1)

        # Calculate triangle edges
        e0 = p1 - p0
        e1 = p2 - p0
        a = np.einsum('...i,...i', e0, e0)
        b = np.einsum('...i,...i', e0, e1)
        c = np.einsum('...i,...i', e1, e1)

        # Calculate determinant and denominator
        det = a * c - b * b
        invDet = 1. / det
        denom = a - 2 * b + c

        # Project to the edges
        p = p0 - points[:, np.newaxis]
        d = np.einsum('...i,...i', e0, p)
        e = np.einsum('...i,...i', e1, p)
        u = b * e - c * d
        v = b * d - a * e

        # Calculate numerators
        bd = b + d
        ce = c + e
        numer0 = (ce - bd) / denom
        numer1 = (c + e - b - d) / denom
        da = -d / a
        ec = -e / c

        # Vectorize test conditions
        m0 = u + v < det
        m1 = u < 0
        m2 = v < 0
        m3 = d < 0
        m4 = (a + d > b + e)
        m5 = ce > bd

        t0 = m0 & m1 & m2 & m3
        t1 = m0 & m1 & m2 & ~m3
        t2 = m0 & m1 & ~m2
        t3 = m0 & ~m1 & m2
        t4 = m0 & ~m1 & ~m2
        t5 = ~m0 & m1 & m5
        t6 = ~m0 & m1 & ~m5
        t7 = ~m0 & m2 & m4
        t8 = ~m0 & m2 & ~m4
        t9 = ~m0 & ~m1 & ~m2

        u = np.where(t0, np.clip(da, 0, 1), u)
        v = np.where(t0, 0, v)
        u = np.where(t1, 0, u)
        v = np.where(t1, 0, v)
        u = np.where(t2, 0, u)
        v = np.where(t2, np.clip(ec, 0, 1), v)
        u = np.where(t3, np.clip(da, 0, 1), u)
        v = np.where(t3, 0, v)
        u *= np.where(t4, invDet, 1)
        v *= np.where(t4, invDet, 1)
        u = np.where(t5, np.clip(numer0, 0, 1), u)
        v = np.where(t5, 1 - u, v)
        u = np.where(t6, 0, u)
        v = np.where(t6, 1, v)
        u = np.where(t7, np.clip(numer1, 0, 1), u)
        v = np.where(t7, 1 - u, v)
        u = np.where(t8, 1, u)
        v = np.where(t8, 0, v)
        u = np.where(t9, np.clip(numer1, 0, 1), u)
        v = np.where(t9, 1 - u, v)

        # Return closest points
        return (p0.T + u[:, np.newaxis] * e0.T + v[:, np.newaxis] * e1.T).swapaxes(2, 1)


def _barycentric_weights(vecs, coords):
    """Compute the weights for barycentric interpolation.

    Parameters
    ----------
    vecs : ndarray of shape (6, 3)
        The 6 vectors used to compute barycentric weights.
        a, e1, e2,
        np.cross(e1, e2),
        np.cross(e2, a),
        np.cross(a, e1)
    coords : ndarray of shape (3, )

    Returns
    -------
    (w, u, v, t) : tuple of float
        ``w``, ``u``, and ``v`` are the weights of the three vertices of the
        triangle, respectively. ``t`` is the scale that needs to be multiplied
        to ``coords`` to make it in the same plane as the three vertices.

    From: https://github.com/neuroboros/neuroboros/blob/f2a2efb914e783add2bf06e0f3715236d3d8550e/src/neuroboros/surface/_barycentric.pyx#L9-L47
    """
    det = coords[0] * vecs[3, 0] + coords[1] * vecs[3, 1] + coords[2] * vecs[3, 2]
    if det == 0:
        if vecs[3, 0] == 0 and vecs[3, 1] == 0 and vecs[3, 2] == 0:
            warnings.warn("Zero cross product of two edges: "
                          "The three vertices are in the same line.")
        else:
            print(vecs[3])
        y = coords - vecs[0]
        u, v = np.linalg.lstsq(vecs[1:3].T, y, rcond=None)[0]
        t = 1.
    else:
        uu = coords[0] * vecs[4, 0] + coords[1] * vecs[4, 1] + coords[2] * vecs[4, 2]
        vv = coords[0] * vecs[5, 0] + coords[1] * vecs[5, 1] + coords[2] * vecs[5, 2]
        u = uu / det
        v = vv / det
        tt = vecs[0, 0] * vecs[3, 0] + vecs[0, 1] * vecs[3, 1] + vecs[0, 2] * vecs[3, 2]
        t = tt / det
    w = 1. - (u + v)
    return w, u, v, t


def _find_close_tris(kdsv, tri_lut, surface):
    tris = []
    for kk in kdsv:
        tris.extend(tri_lut[kk])
    close_tri_verts = surface._triangles[np.unique(tris)]
    close_tris = surface._coords[close_tri_verts]
    return close_tris


def _find_weights(point, close_tris, d_tree):
    point = point[np.newaxis, :]
    tri_dists = cdist(point, _pointsToTriangles(point, close_tris).squeeze())
    closest_tri = close_tris[(tri_dists == tri_dists.min()).squeeze()]
    # make sure a single closest triangle was found
    if closest_tri.shape[0] != 1:
        # in the event of a tie (which can happen)
        # just take the first triangle
        closest_tri = closest_tri[0]

    closest_tri = closest_tri.squeeze()
    # Make sure point is actually inside triangle
    enclosing = True
    if ((point > closest_tri).sum(0) != 3).all():
        enclosing = False
    _, ct_idxs = d_tree.query(closest_tri)
    a = closest_tri[0]
    e1 = closest_tri[1] - a
    e2 = closest_tri[2] - a
    vecs = np.vstack([a, e1, e2, np.cross(e1, e2), np.cross(e2, a), np.cross(a, e1)])
    res = {}
    res[ct_idxs[0]], res[ct_idxs[1]], res[ct_idxs[2]], _ = _barycentric_weights(
        vecs,
        point.squeeze()
    )
    return res, enclosing
