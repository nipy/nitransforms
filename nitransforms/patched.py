import numpy as np
from nibabel.affines import voxel_sizes


def obliquity(affine):
    r"""
    Estimate the *obliquity* an affine's axes represent.
    The term *obliquity* is defined here as the rotation of those axes with
    respect to the cardinal axes.
    This implementation is inspired by `AFNI's implementation
    <https://github.com/afni/afni/blob/b6a9f7a21c1f3231ff09efbd861f8975ad48e525/src/thd_coords.c#L660-L698>`_.
    For further details about *obliquity*, check `AFNI's documentation
    <https://sscc.nimh.nih.gov/sscc/dglen/Obliquity>_.
    Parameters
    ----------
    affine : 2D array-like
        Affine transformation array.  Usually shape (4, 4), but can be any 2D
        array.
    Returns
    -------
    angles : 1D array-like
        The *obliquity* of each axis with respect to the cardinal axes, in radians.
    """
    vs = voxel_sizes(affine)
    best_cosines = np.abs((affine[:-1, :-1] / vs).max(axis=1))
    return np.arccos(best_cosines)


def shape_zoom_affine(shape, zooms, x_flip=True, y_flip=False):
    ''' Get affine implied by given shape and zooms
    We get the translations from the center of the image (implied by
    `shape`).
    Parameters
    ----------
    shape : (N,) array-like
       shape of image data. ``N`` is the number of dimensions
    zooms : (N,) array-like
       zooms (voxel sizes) of the image
    x_flip : {True, False}
       whether to flip the X row of the affine.  Corresponds to
       radiological storage on disk.
    y_flip : {False, True}
       whether to flip the Y row of the affine.  Corresponds to
       DICOM storage on disk when x_flip is also True.
    Returns
    -------
    aff : (4,4) array
       affine giving correspondance of voxel coordinates to mm
       coordinates, taking the center of the image as origin
    Examples
    --------
    >>> shape = (3, 5, 7)
    >>> zooms = (3, 2, 1)
    >>> shape_zoom_affine((3, 5, 7), (3, 2, 1))
    array([[-3.,  0.,  0.,  3.],
           [ 0.,  2.,  0., -4.],
           [ 0.,  0.,  1., -3.],
           [ 0.,  0.,  0.,  1.]])
    >>> shape_zoom_affine((3, 5, 7), (3, 2, 1), False)
    array([[ 3.,  0.,  0., -3.],
           [ 0.,  2.,  0., -4.],
           [ 0.,  0.,  1., -3.],
           [ 0.,  0.,  0.,  1.]])
    '''
    shape = np.asarray(shape)
    zooms = np.array(zooms)  # copy because of flip below
    ndims = len(shape)
    if ndims != len(zooms):
        raise ValueError('Should be same length of zooms and shape')
    if ndims >= 3:
        shape = shape[:3]
        zooms = zooms[:3]
    else:
        full_shape = np.ones((3,))
        full_zooms = np.ones((3,))
        full_shape[:ndims] = shape[:]
        full_zooms[:ndims] = zooms[:]
        shape = full_shape
        zooms = full_zooms
    if x_flip:
        zooms[0] *= -1

    if y_flip:
        zooms[1] *= -1
    # Get translations from center of image
    origin = (shape - 1) / 2.0
    aff = np.eye(4)
    aff[:3, :3] = np.diag(zooms)
    aff[:3, -1] = -origin * zooms
    return aff
