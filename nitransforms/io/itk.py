"""Read/write ITK transforms."""
import numpy as np
from .base import StringBasedStruct


class ITKLinearTransform(StringBasedStruct):
    template_dtype = np.dtype([
        ('type', 'i4'),
        ('id', 'i4'),
        ('parameters', 'f4', (4, 4)),
        ('offset', 'f4', 3),  # Center of rotation
    ])
    dtype = template_dtype

    def __init__(self):
        super().__init__()
        self.structarr['offset'] = [0, 0, 0]
        self.structarr['id'] = 1

    def to_string(self, banner=True):
        sa = self.structarr
        lines = [
            '#Transform {:d}'.format(sa['id']),
            'Transform: MatrixOffsetTransformBase_double_3_3',
            'Parameters: {}'.format(' '.join(
                ['%g' % p
                 for p in sa['parameters'][:3, :3].reshape(-1).tolist() +
                 sa['parameters'][:3, 3].tolist()])),
            'FixedParameters: {:g} {:g} {:g}'.format(*sa['offset']),
            '',
        ]
        if banner:
            lines.insert(0, '#Insight Transform File V1.0')
        return '\n'.join(lines)

    @classmethod
    def from_string(klass, string):
        tf = klass()
        sa = tf.structarr
        lines = [l for l in string.splitlines()
                 if l.strip()]
        assert lines[0][0] == '#'
        if lines[1][0] == '#':
            lines = lines[1:]  # Drop banner with version

        parameters = np.eye(4, dtype='f4')
        sa['id'] = int(lines[0][lines[0].index('T'):].split()[1])
        sa['offset'] = np.genfromtxt([lines[3].split(':')[-1].encode()],
                                     dtype=klass.dtype['offset'])
        vals = np.genfromtxt([lines[2].split(':')[-1].encode()],
                             dtype='f4')
        parameters[:3, :3] = vals[:-3].reshape((3, 3))
        parameters[:3, 3] = vals[-3:]
        sa['parameters'] = parameters
        return tf

    @classmethod
    def from_fileobj(klass, fileobj, check=True):
        return klass.from_string(fileobj.read())


class ITKLinearTransformArray(StringBasedStruct):
    template_dtype = np.dtype([('nxforms', 'i4')])
    dtype = template_dtype
    _xforms = None

    def __init__(self,
                 xforms=None,
                 binaryblock=None,
                 endianness=None,
                 check=True):
        super().__init__(binaryblock, endianness, check)
        self._xforms = []
        for mat in xforms or []:
            xfm = ITKLinearTransform()
            xfm['parameters'] = mat
            self._xforms.append(xfm)

    def __getitem__(self, idx):
        if idx == 'xforms':
            return self._xforms
        if idx == 'nxforms':
            return len(self._xforms)
        return super().__getitem__(idx)

    def to_string(self):
        strings = []
        for i, xfm in enumerate(self._xforms):
            xfm.structarr['id'] = i + 1
            strings.append(xfm.to_string(banner=False))
        strings.insert(0, '#Insight Transform File V1.0')
        return '\n'.join(strings)

    @classmethod
    def from_string(klass, string):
        _self = klass()
        sa = _self.structarr

        lines = [l.strip() for l in string.splitlines()
                 if l.strip()]

        if lines[0][0] != '#' or 'Insight Transform File V1.0' not in lines[0]:
            raise ValueError('Unknown Insight Transform File format.')

        string = '\n'.join(lines[1:])
        for xfm in string.split('#')[1:]:
            _self._xforms.append(ITKLinearTransform.from_string(
                '#%s' % xfm))
        return _self

    @classmethod
    def from_fileobj(klass, fileobj, check=True):
        return klass.from_string(fileobj.read())
