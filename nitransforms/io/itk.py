"""Read/write ITK transforms."""
import warnings
import numpy as np
from scipy.io import savemat as _save_mat
from nibabel import Nifti1Header, Nifti1Image
from nibabel.affines import from_matvec
from .base import (
    BaseLinearTransformList,
    DisplacementsField,
    LinearParameters,
    TransformFileError,
    _read_mat,
)

LPS = np.diag([-1, -1, 1, 1])


class ITKLinearTransform(LinearParameters):
    """A string-based structure for ITK linear transforms."""

    template_dtype = np.dtype([
        ('type', 'i4'),
        ('index', 'i4'),
        ('parameters', 'f4', (4, 4)),
        ('offset', 'f4', 3),  # Center of rotation
    ])
    dtype = template_dtype

    def __init__(self, parameters=None, offset=None):
        """Initialize with default offset and index."""
        super().__init__()
        self.structarr['index'] = 0
        if offset is None:
            offset = np.zeros((3,), dtype='float')
        self.structarr['offset'] = offset
        self.structarr['parameters'] = np.eye(4)
        if parameters is not None:
            self.structarr['parameters'] = parameters

    def __str__(self):
        """Generate a string representation."""
        sa = self.structarr
        lines = [
            '#Transform {:d}'.format(sa['index']),
            'Transform: AffineTransform_float_3_3',
            'Parameters: {}'.format(' '.join(
                ['%g' % p
                 for p in sa['parameters'][:3, :3].reshape(-1).tolist()
                 + sa['parameters'][:3, 3].tolist()])),
            'FixedParameters: {:g} {:g} {:g}'.format(*sa['offset']),
            '',
        ]
        return '\n'.join(lines)

    def to_filename(self, filename):
        """Store this transform to a file with the appropriate format."""
        if str(filename).endswith('.mat'):
            sa = self.structarr
            affine = np.array(np.hstack((
                sa['parameters'][:3, :3].reshape(-1),
                sa['parameters'][:3, 3]))[..., np.newaxis], dtype='f4')
            fixed = np.array(sa['offset'][..., np.newaxis], dtype='f4')
            mdict = {
                'AffineTransform_float_3_3': affine,
                'fixed': fixed,
            }
            _save_mat(str(filename), mdict, format='4')
            return

        with open(str(filename), 'w') as f:
            f.write(self.to_string())

    def to_ras(self, moving=None, reference=None):
        """Return a nitransforms internal RAS+ matrix."""
        sa = self.structarr
        matrix = sa['parameters']
        offset = sa['offset']
        c_neg = from_matvec(np.eye(3), offset * -1.0)
        c_pos = from_matvec(np.eye(3), offset)
        return LPS.dot(c_pos.dot(matrix.dot(c_neg.dot(LPS))))

    def to_string(self, banner=None):
        """Convert to a string directly writeable to file."""
        string = '%s'

        if banner is None:
            banner = self.structarr['index'] == 0

        if banner:
            string = '#Insight Transform File V1.0\n%s'
        return string % self

    @classmethod
    def from_binary(cls, byte_stream, index=0):
        """Read the struct from a matlab binary file."""
        mdict = _read_mat(byte_stream)
        return cls.from_matlab_dict(mdict, index=index)

    @classmethod
    def from_filename(cls, filename):
        """Read the struct from a file given its path."""
        if str(filename).endswith('.mat'):
            with open(str(filename), 'rb') as fileobj:
                return cls.from_binary(fileobj)

        with open(str(filename)) as fileobj:
            return cls.from_string(fileobj.read())

    @classmethod
    def from_fileobj(cls, fileobj, check=True):
        """Read the struct from a file object."""
        if fileobj.name.endswith('.mat'):
            return cls.from_binary(fileobj)
        return cls.from_string(fileobj.read())

    @classmethod
    def from_matlab_dict(cls, mdict, index=0):
        """Read the struct from a matlab dictionary."""
        tf = cls()
        sa = tf.structarr

        sa['index'] = index
        parameters = np.eye(4, dtype='f4')
        parameters[:3, :3] = mdict['AffineTransform_float_3_3'][:-3].reshape((3, 3))
        parameters[:3, 3] = mdict['AffineTransform_float_3_3'][-3:].flatten()
        sa['parameters'] = parameters
        sa['offset'] = mdict['fixed'].flatten()
        return tf

    @classmethod
    def from_ras(cls, ras, index=0, moving=None, reference=None):
        """Create an ITK affine from a nitransform's RAS+ matrix."""
        tf = cls()
        sa = tf.structarr
        sa['index'] = index
        sa['parameters'] = LPS.dot(ras.dot(LPS))
        return tf

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        tf = cls()
        sa = tf.structarr
        lines = [l.strip() for l in string.splitlines()
                 if l.strip()]
        if not lines or not lines[0].startswith('#'):
            raise TransformFileError

        if lines[1][0] == '#':
            lines = lines[1:]  # Drop banner with version

        parameters = np.eye(4, dtype='f4')
        sa['index'] = int(lines[0][lines[0].index('T'):].split()[1])
        sa['offset'] = np.genfromtxt([lines[3].split(':')[-1].encode()],
                                     dtype=cls.dtype['offset'])
        vals = np.genfromtxt([lines[2].split(':')[-1].encode()],
                             dtype='f4')
        parameters[:3, :3] = vals[:-3].reshape((3, 3))
        parameters[:3, 3] = vals[-3:]
        sa['parameters'] = parameters
        return tf


class ITKLinearTransformArray(BaseLinearTransformList):
    """A string-based structure for series of ITK linear transforms."""

    _inner_type = ITKLinearTransform

    @property
    def xforms(self):
        """Get the list of internal ITKLinearTransforms."""
        return self._xforms

    @xforms.setter
    def xforms(self, value):
        self._xforms = list(value)
        for i, val in enumerate(self.xforms):
            val['index'] = i

    def to_filename(self, filename):
        """Store this transform to a file with the appropriate format."""
        if str(filename).endswith('.mat'):
            raise TransformFileError("Please use the ITK's new .h5 format.")

        with open(str(filename), 'w') as f:
            f.write(self.to_string())

    def to_ras(self, moving=None, reference=None):
        """Return a nitransforms' internal RAS matrix."""
        return np.stack([xfm.to_ras() for xfm in self.xforms])

    def to_string(self):
        """Convert to a string directly writeable to file."""
        strings = []
        for i, xfm in enumerate(self.xforms):
            xfm.structarr['index'] = i
            strings.append(xfm.to_string(banner=(i == 0)))
        return '\n'.join(strings)

    @classmethod
    def from_binary(cls, byte_stream):
        """Read the struct from a matlab binary file."""
        raise TransformFileError("Please use the ITK's new .h5 format.")

    @classmethod
    def from_filename(cls, filename):
        """Read the struct from a file given its path."""
        if str(filename).endswith('.mat'):
            with open(str(filename), 'rb') as f:
                return cls.from_binary(f)

        with open(str(filename)) as f:
            string = f.read()
        return cls.from_string(string)

    @classmethod
    def from_fileobj(cls, fileobj, check=True):
        """Read the struct from a file object."""
        if fileobj.name.endswith('.mat'):
            return cls.from_binary(fileobj)
        return cls.from_string(fileobj.read())

    @classmethod
    def from_ras(cls, ras, moving=None, reference=None):
        """
        Create an ITK affine from a nitransform's RAS+ matrix.

        The moving and reference parameters are included in this
        method's signature for a consistent API, but they have no
        effect on this particular method because ITK already uses
        RAS+ coordinates to describe transfroms internally.

        """
        _self = cls()
        _self.xforms = [cls._inner_type.from_ras(ras[i, ...], index=i)
                        for i in range(ras.shape[0])]
        return _self

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        _self = cls()
        lines = [l.strip() for l in string.splitlines()
                 if l.strip()]

        if (
            not lines
            or not lines[0].startswith('#')
            or 'Insight Transform File V1.0' not in lines[0]
        ):
            raise TransformFileError('Unknown Insight Transform File format.')

        string = '\n'.join(lines[1:])
        for xfm in string.split('#')[1:]:
            _self.xforms.append(cls._inner_type.from_string(
                '#%s' % xfm))
        return _self


class ITKDisplacementsField(DisplacementsField):
    """A data structure representing displacements fields."""

    @classmethod
    def from_image(cls, imgobj):
        """Import a displacements field from a NIfTI file."""
        hdr = imgobj.header.copy()
        shape = hdr.get_data_shape()

        if (
            len(shape) != 5
            or shape[-2] != 1
            or not shape[-1] in (2, 3)
        ):
            raise TransformFileError(
                'Displacements field "%s" does not come from ITK.' %
                imgobj.file_map['image'].filename)

        if hdr.get_intent()[0] != 'vector':
            warnings.warn('Incorrect intent identified.')
            hdr.set_intent('vector')

        field = np.squeeze(np.asanyarray(imgobj.dataobj))
        field[..., (0, 1)] *= -1.0

        return imgobj.__class__(field, imgobj.affine, hdr)


class ITKCompositeH5:
    """A data structure for ITK's HDF5 files."""

    @classmethod
    def from_filename(cls, filename):
        """Read the struct from a file given its path."""
        from h5py import File as H5File
        if not str(filename).endswith('.h5'):
            raise RuntimeError("Extension is not .h5")

        with H5File(str(filename)) as f:
            return cls.from_h5obj(f)

    @classmethod
    def from_h5obj(cls, fileobj, check=True):
        """Read the struct from a file object."""
        xfm_list = []
        h5group = fileobj["TransformGroup"]
        typo_fallback = "Transform"
        try:
            h5group['1'][f"{typo_fallback}Parameters"]
        except KeyError:
            typo_fallback = "Tranform"

        for xfm in list(h5group.values())[1:]:
            if xfm["TransformType"][0].startswith(b"AffineTransform"):
                _params = np.asanyarray(xfm[f"{typo_fallback}Parameters"])
                xfm_list.append(
                    ITKLinearTransform(
                        parameters=from_matvec(_params[:-3].reshape(3, 3), _params[-3:]),
                        offset=np.asanyarray(xfm[f"{typo_fallback}FixedParameters"])
                    )
                )
                continue
            if xfm["TransformType"][0].startswith(b"DisplacementFieldTransform"):
                _fixed = np.asanyarray(xfm[f"{typo_fallback}FixedParameters"])
                shape = _fixed[:3].astype('uint16').tolist()
                offset = _fixed[3:6].astype('uint16')
                zooms = _fixed[6:9].astype('float')
                directions = _fixed[9:].astype('float').reshape((3, 3))
                affine = from_matvec(directions * zooms, offset)
                field = np.asanyarray(xfm[f"{typo_fallback}Parameters"]).reshape(
                    tuple(shape + [1, -1])
                )
                hdr = Nifti1Header()
                hdr.set_intent("vector")
                hdr.set_data_dtype("float")

                xfm_list.append(
                    ITKDisplacementsField.from_image(
                        Nifti1Image(field.astype("float"), affine, hdr)
                    )
                )
                continue

            raise NotImplementedError(
                f"Unsupported transform type {xfm['TransformType'][0]}"
            )

        return xfm_list
