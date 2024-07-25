# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Resampling utilities."""
from pathlib import Path
import numpy as np
from nibabel.loadsave import load as _nbload
from nibabel.arrayproxy import get_obj_dtype
from scipy import ndimage as ndi

from nitransforms.linear import Affine, LinearTransformsMapping
from nitransforms.base import (
    ImageGrid,
    TransformError,
    SpatialReference,
    _as_homogeneous,
)

SERIALIZE_VOLUME_WINDOW_WIDTH : int = 8
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

    data = np.asanyarray(spatialimage.dataobj)
    data_nvols = 1 if data.ndim < 4 else data.shape[-1]

    if type(transform) == Affine or type(transform) == LinearTransformsMapping:
        xfm_nvols = len(transform)
    else:
        xfm_nvols = transform.ndim
    """
    if data_nvols == 1 and xfm_nvols > 1:
        data = data[..., np.newaxis]
    elif data_nvols != xfm_nvols:
        raise ValueError(
            "The fourth dimension of the data does not match the transform's shape."
        )
    RESAMPLING FAILS. SUGGEST:
    """
    if data.ndim < transform.ndim:
        data = data[..., np.newaxis]
    elif data_nvols > 1 and data_nvols != xfm_nvols:
        import pdb; pdb.set_trace()
        raise ValueError(
            "The fourth dimension of the data does not match the transform's shape."
        )

    serialize_nvols = serialize_nvols if serialize_nvols and serialize_nvols > 1 else np.inf
    serialize_4d = max(data_nvols, xfm_nvols) > serialize_nvols
    if serialize_4d:
        for t, xfm_t in enumerate(transform):
            ras2vox = ~Affine(spatialimage.affine)
            input_dtype = get_obj_dtype(spatialimage.dataobj)
            output_dtype = output_dtype or input_dtype

            # Map the input coordinates on to timepoint t of the target (moving)
            xcoords = _ref.ndcoords.astype("f4").T
            ycoords = xfm_t.map(xcoords)[..., : _ref.ndim]

            # Calculate corresponding voxel coordinates
            yvoxels = ras2vox.map(ycoords)[..., : _ref.ndim]

            # Interpolate
            dataobj = (
                np.asanyarray(spatialimage.dataobj, dtype=input_dtype)
                if spatialimage.ndim in (2, 3)
                else None
            )
            resampled[..., t] = ndi.map_coordinates(
                (
                    dataobj
                    if dataobj is not None
                    else spatialimage.dataobj[..., t].astype(input_dtype, copy=False)
                ),
                yvoxels.T,
                output=output_dtype,
                order=order,
                mode=mode,
                cval=cval,
                prefilter=prefilter,
            )

    else:
        # For model-based nonlinear transforms, generate the corresponding dense field
        if hasattr(transform, "to_field") and callable(transform.to_field):
            targets = ImageGrid(spatialimage).index(
                _as_homogeneous(
                    transform.to_field(reference=reference).map(_ref.ndcoords.T),
                    dim=_ref.ndim,
                )
            )
        else:
            targets = ImageGrid(spatialimage).index(  # data should be an image
                _as_homogeneous(transform.map(_ref.ndcoords.T), dim=_ref.ndim)
            )

        if transform.ndim == 4:
            targets = _as_homogeneous(targets.reshape(-2, targets.shape[0])).T

        resampled = ndi.map_coordinates(
            data,
            targets,
            output=output_dtype,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
        )

    if isinstance(_ref, ImageGrid):  # If reference is grid, reshape
        hdr = None
        if _ref.header is not None:
            hdr = _ref.header.copy()
            hdr.set_data_dtype(output_dtype or spatialimage.get_data_dtype())
        moved = spatialimage.__class__(
            resampled.reshape(_ref.shape if data.ndim < 4 else _ref.shape + (-1,)),
            _ref.affine,
            hdr,
        )
        return moved

    return resampled
