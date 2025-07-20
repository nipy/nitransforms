# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Common interface for transforms."""
from collections.abc import Iterable
import numpy as np

from .base import (
    TransformBase,
    TransformError,
)
from .linear import Affine, LinearTransformsMapping
from .nonlinear import DenseFieldTransform


class TransformChain(TransformBase):
    """Implements the concatenation of transforms."""

    __slots__ = ("_transforms",)

    def __init__(self, transforms=None):
        """Initialize a chain of transforms."""
        super().__init__()
        self._transforms = None

        if transforms is not None:
            self.transforms = transforms

    def __add__(self, b):
        """
        Compose this and other transforms.

        Example
        -------
        >>> T1 = TransformBase()
        >>> added = T1 + TransformBase() + TransformBase()
        >>> isinstance(added, TransformChain)
        True

        >>> len(added.transforms)
        3

        """
        self.append(b)
        return self

    def __getitem__(self, i):
        """
        Enable indexed access of transform chains.

        Example
        -------
        >>> T1 = TransformBase()
        >>> chain = T1 + TransformBase()
        >>> chain[0] is T1
        True

        """
        return self.transforms[i]

    def __len__(self):
        """Enable using len()."""
        return len(self.transforms)

    @property
    def ndim(self):
        """Get the number of dimensions."""
        return max(x.ndim for x in self._transforms)

    @property
    def transforms(self):
        """Get the internal list of transforms."""
        return self._transforms

    @transforms.setter
    def transforms(self, value):
        self._transforms = _as_chain(value)
        if self.transforms[0].reference:
            self.reference = self.transforms[0].reference

    def append(self, x):
        """
        Concatenate one element to the chain.

        Example
        -------
        >>> chain = TransformChain(transforms=TransformBase())
        >>> chain.append((TransformBase(), TransformBase()))
        >>> len(chain)
        3

        """
        self.transforms += _as_chain(x)

    def insert(self, i, x):
        """
        Insert an item at a given position.

        Example
        -------
        >>> chain = TransformChain(transforms=[TransformBase(), TransformBase()])
        >>> chain.insert(1, TransformBase())
        >>> len(chain)
        3

        >>> chain.insert(1, TransformChain(chain))
        >>> len(chain)
        6

        """
        self.transforms = self.transforms[:i] + _as_chain(x) + self.transforms[i:]

    def map(self, x, inverse=False):
        """
        Apply a succession of transforms, e.g., :math:`y = f_3(f_2(f_1(f_0(x))))`.

        Example
        -------
        >>> chain = TransformChain(transforms=[TransformBase(), TransformBase()])
        >>> chain([(0., 0., 0.), (1., 1., 1.), (-1., -1., -1.)])
        [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0)]

        >>> chain([(0., 0., 0.), (1., 1., 1.), (-1., -1., -1.)], inverse=True)
        [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0)]

        >>> TransformChain()((0., 0., 0.))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        TransformError:

        """
        if not self.transforms:
            raise TransformError("Cannot apply an empty transforms chain.")

        transforms = self.transforms
        if inverse:
            transforms = list(reversed(self.transforms))

        for xfm in transforms:
            x = xfm.map(x, inverse=inverse)

        return x

    def asaffine(self, indices=None):
        """
        Combine a succession of linear transforms into one.

        Example
        ------
        >>> chain = TransformChain(transforms=[
        ...     Affine.from_matvec(vec=(2, -10, 3)),
        ...     Affine.from_matvec(vec=(-2, 10, -3)),
        ... ])
        >>> chain.asaffine()
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])

        >>> chain = TransformChain(transforms=[
        ...     Affine.from_matvec(vec=(1, 2, 3)),
        ...     Affine.from_matvec(mat=[[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        ... ])
        >>> chain.asaffine()
        array([[0., 1., 0., 2.],
               [0., 0., 1., 3.],
               [1., 0., 0., 1.],
               [0., 0., 0., 1.]])

        >>> np.allclose(
        ...     chain.map((4, -2, 1)),
        ...     chain.asaffine().map((4, -2, 1)),
        ... )
        True

        Parameters
        ----------
        indices : :obj:`numpy.array_like`
            The indices of the values to extract.

        """
        affines = self.transforms if indices is None else np.take(self.transforms, indices)
        retval = affines[0]
        for xfm in affines[1:]:
            retval = xfm @ retval
        return retval

    @classmethod
    def from_filename(cls, filename, fmt="X5", reference=None, moving=None, x5_chain=0):
        """Load a transform file."""
        from .io import itk, x5 as x5io
        import h5py
        import nibabel as nb
        from collections import namedtuple

        retval = []
        if str(filename).endswith(".h5") and (fmt is None or fmt.upper() != "X5"):
            reference = None
            xforms = itk.ITKCompositeH5.from_filename(filename)
            for xfmobj in xforms:
                if isinstance(xfmobj, itk.ITKLinearTransform):
                    retval.insert(0, Affine(xfmobj.to_ras(), reference=reference))
                else:
                    retval.insert(0, DenseFieldTransform(xfmobj))

            return TransformChain(retval)

        if fmt and fmt.upper() == "X5":
            with h5py.File(str(filename), "r") as f:
                if f.attrs.get("Format") != "X5":
                    raise TypeError("Input file is not in X5 format")

                tg = [
                    x5io._read_x5_group(node)
                    for _, node in sorted(f["TransformGroup"].items(), key=lambda kv: int(kv[0]))
                ]
                chain_grp = f.get("TransformChain")
                if chain_grp is None:
                    raise TransformError("X5 file contains no TransformChain")

                chain_path = chain_grp[str(x5_chain)][()]
                if isinstance(chain_path, bytes):
                    chain_path = chain_path.decode()
            indices = [int(idx) for idx in chain_path.split("/") if idx]

            Domain = namedtuple("Domain", "affine shape")
            for idx in indices:
                node = tg[idx]
                if node.type == "linear":
                    Transform = Affine if node.array_length == 1 else LinearTransformsMapping
                    reference = None
                    if node.domain is not None:
                        reference = Domain(node.domain.mapping, node.domain.size)
                    retval.append(Transform(node.transform, reference=reference))
                elif node.type == "nonlinear":
                    reference = Domain(node.domain.mapping, node.domain.size)
                    field = nb.Nifti1Image(node.transform, reference.affine)
                    retval.append(
                        DenseFieldTransform(
                            field,
                            is_deltas=node.representation == "displacements",
                            reference=reference,
                        )
                    )
                else:  # pragma: no cover - unsupported type
                    raise NotImplementedError(f"Unsupported transform type {node.type}")

            return TransformChain(retval)

        raise NotImplementedError

    def to_filename(self, filename, fmt="X5"):
        """Store the transform chain in X5 format."""
        from .io import x5 as x5io
        import os
        import h5py

        if fmt.upper() != "X5":
            raise NotImplementedError("Only X5 format is supported for chains")

        if os.path.exists(filename):
            with h5py.File(str(filename), "r") as f:
                existing = [
                    x5io._read_x5_group(node)
                    for _, node in sorted(f["TransformGroup"].items(), key=lambda kv: int(kv[0]))
                ]
        else:
            existing = []

        # convert to objects for equality check
        from collections import namedtuple
        import nibabel as nb

        def _as_transform(x5node):
            Domain = namedtuple("Domain", "affine shape")
            if x5node.type == "linear":
                Transform = Affine if x5node.array_length == 1 else LinearTransformsMapping
                ref = None
                if x5node.domain is not None:
                    ref = Domain(x5node.domain.mapping, x5node.domain.size)
                return Transform(x5node.transform, reference=ref)
            reference = Domain(x5node.domain.mapping, x5node.domain.size)
            field = nb.Nifti1Image(x5node.transform, reference.affine)
            return DenseFieldTransform(
                field,
                is_deltas=x5node.representation == "displacements",
                reference=reference,
            )

        existing_objs = [_as_transform(n) for n in existing]
        path_indices = []
        new_nodes = []
        for xfm in self.transforms:
            # find existing
            idx = None
            for i, obj in enumerate(existing_objs):
                if type(xfm) is type(obj) and xfm == obj:
                    idx = i
                    break
            if idx is None:
                idx = len(existing_objs)
                new_nodes.append((idx, xfm.to_x5()))
                existing_objs.append(xfm)
            path_indices.append(idx)

        mode = "r+" if os.path.exists(filename) else "w"
        with h5py.File(str(filename), mode) as f:
            if "Format" not in f.attrs:
                f.attrs["Format"] = "X5"
                f.attrs["Version"] = np.uint16(1)

            tg = f.require_group("TransformGroup")
            for idx, node in new_nodes:
                g = tg.create_group(str(idx))
                x5io._write_x5_group(g, node)

            cg = f.require_group("TransformChain")
            cg.create_dataset(str(len(cg)), data="/".join(str(i) for i in path_indices))

        return filename


def _as_chain(x):
    """Convert a value into a transform chain."""
    if isinstance(x, TransformChain):
        return x.transforms
    if isinstance(x, TransformBase):
        return [x]
    if isinstance(x, Iterable):
        return list(x)
    return [x]


load = TransformChain.from_filename
