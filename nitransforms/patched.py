import numpy as np
from nibabel.wrapstruct import LabeledWrapStruct as LWS


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


class LabeledWrapStruct(LWS):
    def __setitem__(self, item, value):
        self._structarr[item] = np.asanyarray(value)
