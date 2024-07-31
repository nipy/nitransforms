# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Resampling utilities."""

from functools import partial
from pathlib import Path
import numpy as np
from nibabel.loadsave import load as _nbload
from nibabel.arrayproxy import get_obj_dtype
from scipy import ndimage as ndi

from nitransforms.base import (
    ImageGrid,
    TransformError,
    SpatialReference,
    _as_homogeneous,
)

SERIALIZE_VOLUME_WINDOW_WIDTH: int = 8
"""Minimum number of volumes to automatically serialize 4D transforms."""


def apply(
    transform,
    spatialimage,
    reference=None,
    order=3,
    mode="constant",
    cval=0.0,
    prefilter=True,
    output_dtype=None,
    serialize_nvols=SERIALIZE_VOLUME_WINDOW_WIDTH,
    njobs=None,
):
    """
    Apply a transformation to an image, resampling on the reference spatial object.

    Parameters
    ----------
    spatialimage : `spatialimage`
        The image object containing the data to be resampled in reference
        space
    reference : spatial object, optional
        The image, surface, or combination thereof containing the coordinates
        of samples that will be sampled.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : {'constant', 'reflect', 'nearest', 'mirror', 'wrap'}, optional
        Determines how the input image is extended when the resamplings overflows
        a border. Default is 'constant'.
    cval : float, optional
        Constant value for ``mode='constant'``. Default is 0.0.
    prefilter: bool, optional
        Determines if the image's data array is prefiltered with
        a spline filter before interpolation. The default is ``True``,
        which will create a temporary *float64* array of filtered values
        if *order > 1*. If setting this to ``False``, the output will be
        slightly blurred if *order > 1*, unless the input is prefiltered,
        i.e. it is the result of calling the spline filter on the original
        input.
    output_dtype: dtype specifier, optional
        The dtype of the returned array or image, if specified.
        If ``None``, the default behavior is to use the effective dtype of
        the input image. If slope and/or intercept are defined, the effective
        dtype is float64, otherwise it is equivalent to the input image's
        ``get_data_dtype()`` (on-disk type).
        If ``reference`` is defined, then the return value is an image, with
        a data array of the effective dtype but with the on-disk dtype set to
        the input image's on-disk dtype.

    Returns
    -------
    resampled : `spatialimage` or ndarray
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
    input_dtype = get_obj_dtype(spatialimage.dataobj)

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

        map_coordinates = partial(
            ndi.map_coordinates,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
        )

        def _apply_volume(index, data, transform, targets=None):
            xfm_t = transform if n_resamplings == 1 else transform[index]

            if targets is None:
                targets = ImageGrid(spatialimage).index(  # data should be an image
                    _as_homogeneous(xfm_t.map(ref_ndcoords), dim=_ref.ndim)
                )

            data_t = (
                data
                if data is not None
                else spatialimage.dataobj[..., index].astype(input_dtype, copy=False)
            )
            return map_coordinates(data_t, targets)

        # Order F ensures individual volumes are contiguous in memory
        # Also matches NIfTI, making final save more efficient
        resampled = np.zeros(
            (len(ref_ndcoords), len(transform)), dtype=input_dtype, order="F"
        )
        for t in range(n_resamplings):
            # Interpolate
            resampled[..., t] = _apply_volume(t, data, transform, targets=targets)

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
