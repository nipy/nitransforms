# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Read/write x5 transforms."""

import numpy as np
from h5py import File as H5File


def to_filename(fname, xfm, moving=None):
    """Store the transform in BIDS-Transforms HDF5 file format (.x5)."""
    with H5File(fname, "w") as out_file:
        out_file.attrs["Format"] = "X5"
        out_file.attrs["Version"] = np.uint16(1)
        x5_root = out_file.create_group("/0")

        # Serialize this object into the x5 file format.
        transform_group = x5_root.create_group("TransformGroup")

        # Group '0' containing Affine transform
        transform_0 = transform_group.create_group("0")

        transform_0.attrs["Type"] = "Affine"
        transform_0.create_dataset("Transform", data=xfm.matrix)
        transform_0.create_dataset("Inverse", data=~xfm)

        metadata = {"key": "value"}
        transform_0.attrs["Metadata"] = str(metadata)

        # sub-group 'Domain' contained within group '0'
        transform_0.create_group("Domain")
        # domain_group.attrs["Grid"] = self._grid
        # domain_group.create_dataset("Size", data=_as_homogeneous(self._reference.shape))
        # domain_group.create_dataset("Mapping", data=self.mapping)
