# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Data structures for the X5 transform format.

Implements what's drafted in the `BIDS X5 specification draft
<https://docs.google.com/document/d/1yk5O0QTAOXLdP9iSG3W8ta7IcMFypu2106c3Pnjfi-4/edit>`__.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List

import json
import h5py

import numpy as np


@dataclass
class X5Domain:
    """Domain information of a transform representing reference/moving spaces."""

    grid: bool
    """Whether sampling locations in the manifold are located regularly."""
    size: Sequence[int]
    """The number of sampling locations per dimension (or total if not a grid)."""
    mapping: Optional[np.ndarray]
    """A mapping to go from samples (pixel/voxel coordinates, indices) to space coordinates."""
    coordinates: Optional[str] = None
    """Indexing type of the Mapping field (for example, "cartesian", "spherical" or "index")."""


@dataclass
class X5Transform:
    """Represent one transform entry of an X5 file."""

    type: str
    """A REQUIRED unicode string with possible values: "linear", "nonlinear", "composite"."""
    transform: np.ndarray
    """A REQUIRED array of parameters (e.g., affine matrix, or dense displacements field)."""
    subtype: Optional[str] = None
    """An OPTIONAL extension of type to drive the interpretation of AdditionalParameters."""
    representation: Optional[str] = None
    """
    A string specifiying the transform representation or model, REQUIRED only for nonlinear Type.
    """
    metadata: Optional[Dict[str, Any]] = None
    """An OPTIONAL string (JSON) to embed metadata."""
    dimension_kinds: Optional[Sequence[str]] = None
    """Identifies what "kind" of information is represented by the samples along each axis."""
    domain: Optional[X5Domain] = None
    """
    A dataset specifying the reference manifold for the transform (either
    a regularly gridded 3D space or a surface/sphere).
    REQUIRED for nonlinear Type, RECOMMENDED for linear Type.
    """
    inverse: Optional[np.ndarray] = None
    """Placeholder to pre-calculated inverses."""
    jacobian: Optional[np.ndarray] = None
    """
    A RECOMMENDED data array to keep cached the determinant of Jacobian of the transform
    in case tools have calculated it.
    For parametric models it is generally possible to obtain it analytically, so this dataset
    could not be as useful in that case.
    """
    # additional_parameters: Optional[np.ndarray] = None
    # AdditionalParameters is empty in the draft spec - ignore for now.
    # Only documentation ATM is for SubType:
    # The SubType setting enables setting the additional parameters on a dataset called
    # "AdditionalParameters" that hangs directly from this transform node.
    array_length: int = 1
    """Undocumented field in the draft to enable a single transform group for 4D transforms."""


def to_filename(fname: str | Path, x5_list: List[X5Transform]):
    """
    Write a list of :class:`X5Transform` objects to an X5 HDF5 file.

    Parameters
    ----------
    fname : :obj:`os.PathLike`
        The file name (preferably with the ".x5" extension) in which transforms will be stored.
    x5_list : :obj:`list`
        The list of transforms to be stored in the output dataset.

    Returns
    -------
    fname : :obj:`os.PathLike`
        File containing the transform(s).

    """
    with h5py.File(str(fname), "w") as out_file:
        out_file.attrs["Format"] = "X5"
        out_file.attrs["Version"] = np.uint16(1)
        tg = out_file.create_group("TransformGroup")
        for i, node in enumerate(x5_list):
            g = tg.create_group(str(i))
            g.attrs["Type"] = node.type
            g.attrs["ArrayLength"] = node.array_length
            if node.subtype is not None:
                g.attrs["SubType"] = node.subtype
            if node.representation is not None:
                g.attrs["Representation"] = node.representation
            if node.metadata is not None:
                g.attrs["Metadata"] = json.dumps(node.metadata)
            g.create_dataset("Transform", data=node.transform)
            g.create_dataset(
                "DimensionKinds",
                data=np.asarray(node.dimension_kinds, dtype="S"),
            )
            if node.domain is not None:
                dgrp = g.create_group("Domain")
                dgrp.create_dataset("Grid", data=np.uint8(1 if node.domain.grid else 0))
                dgrp.create_dataset("Size", data=np.asarray(node.domain.size))
                dgrp.create_dataset("Mapping", data=node.domain.mapping)
                if node.domain.coordinates is not None:
                    dgrp.attrs["Coordinates"] = node.domain.coordinates

            if node.inverse is not None:
                g.create_dataset("Inverse", data=node.inverse)
            if node.jacobian is not None:
                g.create_dataset("Jacobian", data=node.jacobian)
            # Disabled until we need SubType and AdditionalParameters
            # if node.additional_parameters is not None:
            #     g.create_dataset(
            #         "AdditionalParameters", data=node.additional_parameters
            #     )
    return fname
