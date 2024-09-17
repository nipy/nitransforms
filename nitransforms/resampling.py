# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Resampling utilities."""

import asyncio
from os import cpu_count
from functools import partial
from pathlib import Path
from typing import Callable, TypeVar, Union

import numpy as np
from nibabel.loadsave import load as _nbload
from nibabel.arrayproxy import get_obj_dtype
from nibabel.spatialimages import SpatialImage
from scipy import ndimage as ndi

from nitransforms.base import (
    ImageGrid,
    TransformBase,
    TransformError,
    SpatialReference,
    _as_homogeneous,
)

R = TypeVar("R")

SERIALIZE_VOLUME_WINDOW_WIDTH: int = 8
"""Minimum number of volumes to automatically serialize 4D transforms."""


async def worker(job: Callable[[], R], semaphore) -> R:
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, job)


async def _apply_serial(
    data: np.ndarray,
    spatialimage: SpatialImage,
    targets: np.ndarray,
    transform: TransformBase,
    ref_ndim: int,
    ref_ndcoords: np.ndarray,
    n_resamplings: int,
    output: np.ndarray,
    input_dtype: np.dtype,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
    max_concurrent: int = min(cpu_count(), 12),
):
    """
    Resample through a given transform serially, in a 3D+t setting.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        The input data array.
    spatialimage : :obj:`~nibabel.spatialimages.SpatialImage` or `os.pathlike`
        The image object containing the data to be resampled in reference
        space
    targets : :obj:`~numpy.ndarray`
        The target coordinates for mapping.
    transform : :obj:`~nitransforms.base.TransformBase`
        The 3D, 3D+t, or 4D transform through which data will be resampled.
    ref_ndim : :obj:`int`
        Dimensionality of the resampling target (reference image).
    ref_ndcoords : :obj:`~numpy.ndarray`
        Physical coordinates (RAS+) where data will be interpolated, if the resampling
        target is a grid, the scanner coordinates of all voxels.
    n_resamplings : :obj:`int`
        Total number of 3D resamplings (can be defined by the input image, the transform,
        or be matched, that is, same number of volumes in the input and number of transforms).
    output : :obj:`~numpy.ndarray`
        The output data array where resampled values will be stored volume-by-volume.
    order : :obj:`int`, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : :obj:`str`, optional
        Determines how the input image is extended when the resamplings overflows
        a border. One of ``'constant'``, ``'reflect'``, ``'nearest'``, ``'mirror'``,
        or ``'wrap'``. Default is ``'constant'``.
    cval : :obj:`float`, optional
        Constant value for ``mode='constant'``. Default is 0.0.
    prefilter: :obj:`bool`, optional
        Determines if the image's data array is prefiltered with
        a spline filter before interpolation. The default is ``True``,
        which will create a temporary *float64* array of filtered values
        if *order > 1*. If setting this to ``False``, the output will be
        slightly blurred if *order > 1*, unless the input is prefiltered,
        i.e. it is the result of calling the spline filter on the original
        input.

    Returns
    -------
    np.ndarray
        Data resampled on the 3D+t array of input coordinates.

    """
    tasks = []
    semaphore = asyncio.Semaphore(max_concurrent)

    for t in range(n_resamplings):
        xfm_t = transform if n_resamplings == 1 else transform[t]

        if targets is None:
            targets = ImageGrid(spatialimage).index(  # data should be an image
                _as_homogeneous(xfm_t.map(ref_ndcoords), dim=ref_ndim)
            )

        data_t = (
            data
            if data is not None
            else spatialimage.dataobj[..., t].astype(input_dtype, copy=False)
        )

        tasks.append(
            asyncio.create_task(
                worker(
                    partial(
                        ndi.map_coordinates,
                        data_t,
                        targets,
                        output=output[..., t],
                        order=order,
                        mode=mode,
                        cval=cval,
                        prefilter=prefilter,
                    ),
                    semaphore,
                )
            )
        )
    await asyncio.gather(*tasks)
    return output


def apply(
    transform: TransformBase,
    spatialimage: Union[str, Path, SpatialImage],
    reference: Union[str, Path, SpatialImage] = None,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
    output_dtype: np.dtype = None,
    dtype_width: int = 8,
    serialize_nvols: int = SERIALIZE_VOLUME_WINDOW_WIDTH,
    max_concurrent: int = min(cpu_count(), 12),
) -> Union[SpatialImage, np.ndarray]:
    """
    Apply a transformation to an image, resampling on the reference spatial object.

    Parameters
    ----------
    transform: :obj:`~nitransforms.base.TransformBase`
        The 3D, 3D+t, or 4D transform through which data will be resampled.
    spatialimage : :obj:`~nibabel.spatialimages.SpatialImage` or `os.pathlike`
        The image object containing the data to be resampled in reference
        space
    reference : :obj:`~nibabel.spatialimages.SpatialImage` or `os.pathlike`
        The image, surface, or combination thereof containing the coordinates
        of samples that will be sampled.
    order : :obj:`int`, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : :obj:`str`, optional
        Determines how the input image is extended when the resamplings overflows
        a border. One of ``'constant'``, ``'reflect'``, ``'nearest'``, ``'mirror'``,
        or ``'wrap'``. Default is ``'constant'``.
    cval : :obj:`float`, optional
        Constant value for ``mode='constant'``. Default is 0.0.
    prefilter : :obj:`bool`, optional
        Determines if the image's data array is prefiltered with
        a spline filter before interpolation. The default is ``True``,
        which will create a temporary *float64* array of filtered values
        if *order > 1*. If setting this to ``False``, the output will be
        slightly blurred if *order > 1*, unless the input is prefiltered,
        i.e. it is the result of calling the spline filter on the original
        input.
    output_dtype : :obj:`~numpy.dtype`, optional
        The dtype of the returned array or image, if specified.
        If ``None``, the default behavior is to use the effective dtype of
        the input image. If slope and/or intercept are defined, the effective
        dtype is float64, otherwise it is equivalent to the input image's
        ``get_data_dtype()`` (on-disk type).
        If ``reference`` is defined, then the return value is an image, with
        a data array of the effective dtype but with the on-disk dtype set to
        the input image's on-disk dtype.
    dtype_width : :obj:`int`
        Cap the width of the input data type to the given number of bytes.
        This argument is intended to work as a way to implement lower memory
        requirements in resampling.
    serialize_nvols : :obj:`int`
        Minimum number of volumes in a 3D+t (that is, a series of 3D transformations
        independent in time) to resample on a one-by-one basis.
        Serialized resampling can be executed concurrently (parallelized) with
        the argument ``max_concurrent``.
    max_concurrent : :obj:`int`
        Maximum number of 3D resamplings to be executed concurrently.

    Returns
    -------
    resampled : :obj:`~nibabel.spatialimages.SpatialImage` or :obj:`~numpy.ndarray`
        The data imaged after resampling to reference space.

    """
    if reference is not None and isinstance(reference, (str, Path)):
        reference = _nbload(str(reference))

    _ref = (
        transform.reference
        if reference is None
        else SpatialReference.factory(reference)
    )

    if _ref is None:
        raise TransformError("Cannot apply transform without reference")

    if isinstance(spatialimage, (str, Path)):
        spatialimage = _nbload(str(spatialimage))

    # Avoid opening the data array just yet
    input_dtype = cap_dtype(get_obj_dtype(spatialimage.dataobj), dtype_width)

    # Number of data volumes
    data_nvols = 1 if spatialimage.ndim < 4 else spatialimage.shape[-1]
    # Number of transforms: transforms chains (e.g., affine + field, are a single transform)
    xfm_nvols = 1 if transform.ndim < 4 else len(transform)

    if data_nvols != xfm_nvols and min(data_nvols, xfm_nvols) > 1:
        raise ValueError(
            "The fourth dimension of the data does not match the transform's shape."
        )

    serialize_nvols = (
        serialize_nvols if serialize_nvols and serialize_nvols > 1 else np.inf
    )
    n_resamplings = max(data_nvols, xfm_nvols)
    serialize_4d = n_resamplings >= serialize_nvols

    targets = None
    ref_ndcoords = _ref.ndcoords.T
    if hasattr(transform, "to_field") and callable(transform.to_field):
        targets = ImageGrid(spatialimage).index(
            _as_homogeneous(
                transform.to_field(reference=reference).map(ref_ndcoords),
                dim=_ref.ndim,
            )
        )
    elif xfm_nvols == 1:
        targets = ImageGrid(spatialimage).index(  # data should be an image
            _as_homogeneous(transform.map(ref_ndcoords), dim=_ref.ndim)
        )

    if serialize_4d:
        data = (
            np.asanyarray(spatialimage.dataobj, dtype=input_dtype)
            if data_nvols == 1
            else None
        )

        # Order F ensures individual volumes are contiguous in memory
        # Also matches NIfTI, making final save more efficient
        resampled = np.zeros(
            (len(ref_ndcoords), len(transform)), dtype=input_dtype, order="F"
        )

        resampled = asyncio.run(
            _apply_serial(
                data,
                spatialimage,
                targets,
                transform,
                _ref.ndim,
                ref_ndcoords,
                n_resamplings,
                resampled,
                input_dtype,
                order=order,
                mode=mode,
                cval=cval,
                prefilter=prefilter,
                max_concurrent=max_concurrent,
            )
        )
    else:
        data = np.asanyarray(spatialimage.dataobj, dtype=input_dtype)

        if targets is None:
            targets = ImageGrid(spatialimage).index(  # data should be an image
                _as_homogeneous(transform.map(ref_ndcoords), dim=_ref.ndim)
            )

        # Cast 3D data into 4D if 4D nonsequential transform
        if data_nvols == 1 and xfm_nvols > 1:
            data = data[..., np.newaxis]

        if transform.ndim == 4:
            targets = _as_homogeneous(targets.reshape(-2, targets.shape[0])).T

        resampled = ndi.map_coordinates(
            data,
            targets,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
        )

    if isinstance(_ref, ImageGrid):  # If reference is grid, reshape
        hdr = (
            _ref.header.copy()
            if _ref.header is not None
            else spatialimage.header.__class__()
        )
        hdr.set_data_dtype(output_dtype or spatialimage.header.get_data_dtype())

        moved = spatialimage.__class__(
            resampled.reshape(_ref.shape if n_resamplings == 1 else _ref.shape + (-1,)),
            _ref.affine,
            hdr,
        )
        return moved

    output_dtype = output_dtype or input_dtype
    return resampled.astype(output_dtype)


def cap_dtype(dt, nbytes):
    """
    Cap the datatype size to shave off memory requirements.

    Examples
    --------
    >>> cap_dtype(np.dtype('f8'), 4)
    dtype('float32')

    >>> cap_dtype(np.dtype('f8'), 16)
    dtype('float64')

    >>> cap_dtype('float64', 4)
    dtype('float32')

    >>> cap_dtype(np.dtype('i1'), 4)
    dtype('int8')

    >>> cap_dtype('int8', 4)
    dtype('int8')

    >>> cap_dtype('int32', 1)
    dtype('int8')

    >>> cap_dtype(np.dtype('i8'), 4)
    dtype('int32')

    """
    dt = np.dtype(dt)
    return np.dtype(f"{dt.byteorder}{dt.kind}{min(nbytes, dt.itemsize)}")
