# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Read and write transforms."""

from nitransforms.io import afni, fsl, itk, lta, x5
from nitransforms.io.base import TransformIOError, TransformFileError

__all__ = [
    "afni",
    "fsl",
    "itk",
    "lta",
    "x5",
    "get_linear_factory",
    "TransformFileError",
    "TransformIOError",
]

_IO_TYPES = {
    "itk": (itk, "ITKLinearTransform"),
    "ants": (itk, "ITKLinearTransform"),
    "elastix": (itk, "ITKLinearTransform"),
    "lta": (lta, "FSLinearTransform"),
    "fs": (lta, "FSLinearTransform"),
    "fsl": (fsl, "FSLLinearTransform"),
    "afni": (afni, "AFNILinearTransform"),
}


def get_linear_factory(fmt, is_array=True):
    """
    Return the type required by a given format.

    Parameters
    ----------
    fmt : :obj:`str`
        A format identifying string.
    is_array : :obj:`bool`
        Whether the array version of the class should be returned.

    Returns
    -------
    type
        The class object (not an instance) of the linear transfrom to be created
        (for example, :obj:`~nitransforms.io.itk.ITKLinearTransform`).

    Examples
    --------
    >>> get_linear_factory("itk")
    <class 'nitransforms.io.itk.ITKLinearTransformArray'>
    >>> get_linear_factory("itk", is_array=False)
    <class 'nitransforms.io.itk.ITKLinearTransform'>
    >>> get_linear_factory("fsl")
    <class 'nitransforms.io.fsl.FSLLinearTransformArray'>
    >>> get_linear_factory("fsl", is_array=False)
    <class 'nitransforms.io.fsl.FSLLinearTransform'>
    >>> get_linear_factory("fakepackage")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    TypeError: Unsupported transform format <fakepackage>.

    """
    if fmt.lower() not in _IO_TYPES:
        raise TypeError(f"Unsupported transform format <{fmt}>.")

    module, classname = _IO_TYPES[fmt.lower()]
    return getattr(module, f"{classname}{'Array' * is_array}")
