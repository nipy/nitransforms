"""Read/write linear transforms."""
import numpy as np
from nibabel.volumeutils import Recoder
from nibabel.affines import voxel_sizes

from .base import BaseLinearTransformList, StringBasedStruct, TransformFileError


transform_codes = Recoder((
    (0, 'LINEAR_VOX_TO_VOX'),
    (1, 'LINEAR_RAS_TO_RAS'),
    (2, 'LINEAR_PHYSVOX_TO_PHYSVOX'),
    (14, 'REGISTER_DAT'),
    (21, 'LINEAR_COR_TO_COR')),
    fields=('code', 'label'))


class VolumeGeometry(StringBasedStruct):
    """Data structure for regularly gridded images."""

    template_dtype = np.dtype([
        ('valid', 'i4'),              # Valid values: 0, 1
        ('volume', 'i4', (3, 1)),     # width, height, depth
        ('voxelsize', 'f4', (3, 1)),  # xsize, ysize, zsize
        ('xras', 'f8', (3, 1)),       # x_r, x_a, x_s
        ('yras', 'f8', (3, 1)),       # y_r, y_a, y_s
        ('zras', 'f8', (3, 1)),       # z_r, z_a, z_s
        ('cras', 'f8', (3, 1)),       # c_r, c_a, c_s
        ('filename', 'U1024')])       # Not conformant (may be >1024 bytes)
    dtype = template_dtype

    def as_affine(self):
        """Return the internal affine of this regular grid."""
        affine = np.eye(4)
        sa = self.structarr
        A = np.hstack((sa['xras'], sa['yras'], sa['zras'])) * sa['voxelsize']
        b = sa['cras'] - A.dot(sa['volume']) / 2
        affine[:3, :3] = A
        affine[:3, [3]] = b
        return affine

    def __str__(self):
        """Format the structure as a text file."""
        sa = self.structarr
        lines = [
            'valid = {}  # volume info {:s}valid'.format(
                sa['valid'], '' if sa['valid'] else 'in'),
            'filename = {}'.format(sa['filename']),
            'volume = {:d} {:d} {:d}'.format(*sa['volume'].flatten()),
            'voxelsize = {:.15e} {:.15e} {:.15e}'.format(
                *sa['voxelsize'].flatten()),
            'xras   = {:.15e} {:.15e} {:.15e}'.format(*sa['xras'].flatten()),
            'yras   = {:.15e} {:.15e} {:.15e}'.format(*sa['yras'].flatten()),
            'zras   = {:.15e} {:.15e} {:.15e}'.format(*sa['zras'].flatten()),
            'cras   = {:.15e} {:.15e} {:.15e}'.format(*sa['cras'].flatten()),
        ]
        return '\n'.join(lines)

    def to_string(self):
        """Format the structure as a text file."""
        return self.__str__()

    @classmethod
    def from_image(klass, img):
        """Create struct from an image."""
        volgeom = klass()
        sa = volgeom.structarr
        sa['valid'] = 1
        sa['volume'][:, 0] = img.shape[:3]    # Assumes xyzt-ordered image
        sa['voxelsize'][:, 0] = voxel_sizes(img.affine)[:3]
        A = img.affine[:3, :3]
        b = img.affine[:3, [3]]
        cols = A / sa['voxelsize']
        sa['xras'] = cols[:, [0]]
        sa['yras'] = cols[:, [1]]
        sa['zras'] = cols[:, [2]]
        sa['cras'] = b + A.dot(sa['volume']) / 2
        try:
            sa['filename'] = img.file_map['image'].filename
        except Exception:
            pass

        return volgeom

    @classmethod
    def from_string(klass, string):
        """Create a volume structure off of text."""
        volgeom = klass()
        sa = volgeom.structarr
        lines = string.splitlines()
        for key in ('valid', 'filename', 'volume', 'voxelsize',
                    'xras', 'yras', 'zras', 'cras'):
            label, valstring = lines.pop(0).split(' =')
            assert label.strip() == key

            val = ''
            if valstring.strip():
                parsed = np.genfromtxt([valstring.encode()], autostrip=True,
                                       dtype=klass.dtype[key])
                if parsed.size:
                    val = parsed.reshape(sa[key].shape)
            sa[key] = val
        return volgeom


class LinearTransform(StringBasedStruct):
    """Represents a single LTA's transform structure."""

    template_dtype = np.dtype([
        ('type', 'i4'),
        ('mean', 'f4', (3, 1)),  # x0, y0, z0
        ('sigma', 'f4'),
        ('m_L', 'f8', (4, 4)),
        ('m_dL', 'f8', (4, 4)),
        ('m_last_dL', 'f8', (4, 4)),
        ('src', VolumeGeometry),
        ('dst', VolumeGeometry),
        ('label', 'i4')])
    dtype = template_dtype

    def __getitem__(self, idx):
        """Implement dictionary access."""
        val = super(LinearTransform, self).__getitem__(idx)
        if idx in ('src', 'dst'):
            val = VolumeGeometry(val)
        return val

    def set_type(self, new_type):
        """
        Convert the internal transformation matrix to a different type inplace.

        Parameters
        ----------
        new_type : str, int
            Tranformation type

        """
        sa = self.structarr
        src = VolumeGeometry(sa['src'])
        dst = VolumeGeometry(sa['dst'])
        current = sa['type']
        if isinstance(new_type, str):
            new_type = transform_codes.code[new_type]

        if current == new_type:
            return

        # VOX2VOX -> RAS2RAS
        if (current, new_type) == (0, 1):
            M = dst.as_affine().dot(sa['m_L'].dot(np.linalg.inv(src.as_affine())))
            sa['m_L'] = M
            sa['type'] = new_type
            return

        raise NotImplementedError(
            "Converting {0} to {1} is not yet available".format(
                transform_codes.label[current],
                transform_codes.label[new_type]
            )
        )

    def to_ras(self, moving=None, reference=None):
        """
        Return a nitransforms' internal RAS+ array.

        Seemingly, the matrix of an LTA is defined such that it
        maps coordinates from the ``dest volume`` to the ``src volume``.
        Therefore, without inversion, the LTA matrix is appropiate
        to move the information from ``src volume`` into the
        ``dest volume``'s grid.

        .. important ::

            The ``moving`` and ``reference`` parameters are dismissed
            because ``VOX2VOX`` LTAs are converted to ``RAS2RAS`` type
            before returning the RAS+ matrix, using the ``dest`` and
            ``src`` contained in the LTA. Both arguments are kept for
            API compatibility.

        Parameters
        ----------
        moving : dismissed
            The spatial reference of moving images.
        reference : dismissed
            The spatial reference of moving images.

        Returns
        -------
        matrix : :obj:`numpy.ndarray`
            The RAS+ affine matrix corresponding to the LTA.

        """
        self.set_type(1)
        return np.linalg.inv(self.structarr['m_L'])

    def to_string(self):
        """Convert this transform to text."""
        sa = self.structarr
        lines = [
            'mean      = {:6.4f} {:6.4f} {:6.4f}'.format(
                *sa['mean'].flatten()),
            'sigma     = {:6.4f}'.format(float(sa['sigma'])),
            '1 4 4',
            ('{:18.15e} ' * 4).format(*sa['m_L'][0]),
            ('{:18.15e} ' * 4).format(*sa['m_L'][1]),
            ('{:18.15e} ' * 4).format(*sa['m_L'][2]),
            ('{:18.15e} ' * 4).format(*sa['m_L'][3]),
            'src volume info',
            str(self['src']),
            'dst volume info',
            str(self['dst']),
        ]
        return '\n'.join(lines)

    @classmethod
    def from_string(klass, string):
        """Read a transform from text."""
        lt = klass()
        sa = lt.structarr
        lines = string.splitlines()
        for key in ('mean', 'sigma'):
            label, valstring = lines.pop(0).split(' = ')
            assert label.strip() == key

            val = np.genfromtxt([valstring.encode()],
                                dtype=klass.dtype[key])
            sa[key] = val.reshape(sa[key].shape)
        assert lines.pop(0) == '1 4 4'  # xforms, shape + 1, shape + 1
        val = np.genfromtxt([valstring.encode() for valstring in lines[:4]],
                            dtype='f4')
        sa['m_L'] = val
        lines = lines[4:]
        assert lines.pop(0) == 'src volume info'
        sa['src'] = np.asanyarray(VolumeGeometry.from_string('\n'.join(lines[:8])))
        lines = lines[8:]
        assert lines.pop(0) == 'dst volume info'
        sa['dst'] = np.asanyarray(VolumeGeometry.from_string('\n'.join(lines)))
        return lt


class LinearTransformArray(BaseLinearTransformList):
    """A list of linear transforms generated by FreeSurfer."""

    template_dtype = np.dtype([
        ('type', 'i4'),
        ('nxforms', 'i4'),
        ('subject', 'U1024'),
        ('fscale', 'f4')])
    dtype = template_dtype
    _inner_type = LinearTransform

    def __getitem__(self, idx):
        """Allow dictionary access to the transforms."""
        if idx == 'xforms':
            return self._xforms
        if idx == 'nxforms':
            return len(self._xforms)
        return self.structarr[idx]

    def to_ras(self, moving=None, reference=None):
        """Set type to RAS2RAS and return the new matrix."""
        self.structarr['type'] = 1
        return [xfm.to_ras() for xfm in self.xforms]

    def to_string(self):
        """Convert this LTA into text format."""
        code = int(self['type'])
        header = [
            'type      = {} # {}'.format(code, transform_codes.label[code]),
            'nxforms   = {}'.format(self['nxforms'])]
        xforms = [xfm.to_string() for xfm in self._xforms]
        footer = [
            'subject {}'.format(self['subject']),
            'fscale {:.6f}'.format(float(self['fscale']))]
        return '\n'.join(header + xforms + footer)

    @classmethod
    def from_string(klass, string):
        """Read this LTA from a text string."""
        lta = klass()
        sa = lta.structarr
        lines = [l.strip() for l in string.splitlines()
                 if l.strip() and not l.strip().startswith('#')]
        if not lines or not lines[0].startswith('type'):
            raise TransformFileError("Invalid LTA format")
        for key in ('type', 'nxforms'):
            label, valstring = lines.pop(0).split(' = ')
            assert label.strip() == key

            val = np.genfromtxt([valstring.encode()],
                                dtype=klass.dtype[key])
            sa[key] = val.reshape(sa[key].shape) if val.size else ''
        for _ in range(sa['nxforms']):
            lta._xforms.append(
                klass._inner_type.from_string('\n'.join(lines[:25])))
            lta._xforms[-1].structarr['type'] = sa['type']
            lines = lines[25:]
        for key in ('subject', 'fscale'):
            # Optional keys
            if not (lines and lines[0].startswith(key)):
                continue
            try:
                label, valstring = lines.pop(0).split(' ')
            except ValueError:
                sa[key] = ''
            else:
                assert label.strip() == key

                val = np.genfromtxt([valstring.encode()],
                                    dtype=klass.dtype[key])
                sa[key] = val.reshape(sa[key].shape) if val.size else ''

        assert len(lta._xforms) == sa['nxforms']
        return lta
