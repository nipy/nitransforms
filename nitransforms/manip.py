# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Common interface for transforms."""

import os
from collections.abc import Iterable
import numpy as np

import h5py
from nitransforms.base import (
    TransformBase,
    TransformError,
)
from nitransforms.io import itk, x5 as x5io
from nitransforms.io.x5 import from_filename as load_x5
from nitransforms.linear import (
    Affine,
    from_x5 as linear_from_x5,  # noqa: F401
)
from nitransforms.nonlinear import (
    DenseFieldTransform,
    from_x5 as nonlinear_from_x5,  # noqa: F401
)


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
        affines = (
            self.transforms if indices is None else np.take(self.transforms, indices)
        )
        retval = affines[0]
        for xfm in affines[1:]:
            retval = xfm @ retval
        return retval

    @classmethod
    def from_filename(cls, filename, fmt="X5", reference=None, moving=None, x5_chain=0):
        """Load a transform file."""

        retval = []
        if fmt and fmt.upper() == "X5":
            # Get list of X5 nodes and generate transforms
            xfm_list = [
                globals()[f"{node.type}_from_x5"]([node]) for node in load_x5(filename)
            ]
            if not xfm_list:
                raise TransformError("Empty transform group")

            if x5_chain is None:
                return xfm_list

            with h5py.File(str(filename), "r") as f:
                chain_grp = f.get("TransformChain")
                if chain_grp is None:
                    raise TransformError("X5 file contains no TransformChain")

                chain_path = chain_grp[str(x5_chain)][()]
                if isinstance(chain_path, bytes):
                    chain_path = chain_path.decode()

            return TransformChain([xfm_list[int(idx)] for idx in chain_path.split("/")])

        if str(filename).endswith(".h5"):
            reference = None
            xforms = itk.ITKCompositeH5.from_filename(filename)
            for xfmobj in xforms:
                if isinstance(xfmobj, itk.ITKLinearTransform):
                    retval.insert(0, Affine(xfmobj.to_ras(), reference=reference))
                else:
                    retval.insert(0, DenseFieldTransform(xfmobj))

            return TransformChain(retval)

        raise NotImplementedError

    def to_filename(self, filename, fmt="X5"):
        """Store the transform chain in X5 format."""

        if fmt.upper() != "X5":
            raise NotImplementedError("Only X5 format is supported for chains")

        existing = (
            self.from_filename(filename, x5_chain=None)
            if os.path.exists(filename)
            else []
        )

        xfm_chain = []
        new_xfms = []
        next_xfm_index = len(existing)
        for xfm in self.transforms:
            for eidx, existing_xfm in enumerate(existing):
                if xfm == existing_xfm:
                    xfm_chain.append(eidx)
                    break
            else:
                xfm_chain.append(next_xfm_index)
                new_xfms.append((next_xfm_index, xfm))
                existing.append(xfm)
                next_xfm_index += 1

        mode = "r+" if os.path.exists(filename) else "w"
        with h5py.File(str(filename), mode) as f:
            if "Format" not in f.attrs:
                f.attrs["Format"] = "X5"
                f.attrs["Version"] = np.uint16(1)

            tg = f.require_group("TransformGroup")
            for idx, node in new_xfms:
                g = tg.create_group(str(idx))
                x5io._write_x5_group(g, node.to_x5())

            cg = f.require_group("TransformChain")
            cg.create_dataset(str(len(cg)), data="/".join(str(i) for i in xfm_chain))

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
