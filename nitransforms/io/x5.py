"""Data structures for the X5 transform format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, List

import json
import h5py

import numpy as np


@dataclass
class X5Domain:
    """Domain information of a transform."""

    grid: bool
    size: Sequence[int]
    mapping: np.ndarray
    coordinates: Optional[str] = None


@dataclass
class X5Transform:
    """Represent one transform entry of an X5 file."""

    type: str
    transform: np.ndarray
    dimension_kinds: Sequence[str]
    array_length: int = 1
    domain: Optional[X5Domain] = None
    subtype: Optional[str] = None
    representation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    inverse: Optional[np.ndarray] = None
    jacobian: Optional[np.ndarray] = None
    additional_parameters: Optional[np.ndarray] = None


def to_filename(fname: str, x5_list: List[X5Transform]):
    """Write a list of :class:`X5Transform` objects to an X5 HDF5 file."""
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
            if node.additional_parameters is not None:
                g.create_dataset(
                    "AdditionalParameters", data=node.additional_parameters
                )
    return fname
