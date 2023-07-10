"""Read/write linear transforms."""
import numpy as np
from nibabel.volumeutils import Recoder
from nibabel.affines import voxel_sizes, from_matvec

from .base import (
    BaseLinearTransformList,
    StringBasedStruct,
    LinearTransformStruct,
    TransformFileError,
)


transform_codes = Recoder(
    (
        (0, "LINEAR_VOX_TO_VOX"),
        (1, "LINEAR_RAS_TO_RAS"),
        (2, "LINEAR_PHYSVOX_TO_PHYSVOX"),
        (14, "REGISTER_DAT"),
        (21, "LINEAR_COR_TO_COR"),
    ),
    fields=("code", "label"),
)


class VolumeGeometry(StringBasedStruct):
    """Data structure for regularly gridded images."""

    template_dtype = np.dtype(
        [
            ("valid", "i4"),  # Valid values: 0, 1
            ("volume", "i4", (3, )),  # width, height, depth
            ("voxelsize", "f4", (3, )),  # xsize, ysize, zsize
            ("xras", "f8", (3, 1)),  # x_r, x_a, x_s
            ("yras", "f8", (3, 1)),  # y_r, y_a, y_s
            ("zras", "f8", (3, 1)),  # z_r, z_a, z_s
            ("cras", "f8", (3, )),  # c_r, c_a, c_s
            ("filename", "U1024"),
        ]
    )  # Not conformant (may be >1024 bytes)
    dtype = template_dtype

    def as_affine(self):
        """Return the internal affine of this regular grid."""
        sa = self.structarr
        A = np.hstack((sa["xras"], sa["yras"], sa["zras"])) * sa["voxelsize"]
        b = sa["cras"] - A @ sa["volume"] / 2
        return from_matvec(A, b)

    def __str__(self):
        """Format the structure as a text file."""
        sa = self.structarr
        lines = [
            "valid = {}  # volume info {:s}valid".format(
                sa["valid"], "" if sa["valid"] else "in"
            ),
            "filename = {}".format(sa["filename"]),
            "volume = {:d} {:d} {:d}".format(*sa["volume"]),
            "voxelsize = {:.15e} {:.15e} {:.15e}".format(*sa["voxelsize"]),
            "xras   = {:.15e} {:.15e} {:.15e}".format(*sa["xras"].flatten()),
            "yras   = {:.15e} {:.15e} {:.15e}".format(*sa["yras"].flatten()),
            "zras   = {:.15e} {:.15e} {:.15e}".format(*sa["zras"].flatten()),
            "cras   = {:.15e} {:.15e} {:.15e}".format(*sa["cras"].flatten()),
        ]
        return "\n".join(lines)

    def to_string(self):
        """Format the structure as a text file."""
        return self.__str__()

    @classmethod
    def from_image(cls, img):
        """Create struct from an image."""
        volgeom = cls()
        sa = volgeom.structarr
        sa["valid"] = 1
        sa["volume"] = img.shape[:3]  # Assumes xyzt-ordered image
        sa["voxelsize"] = voxel_sizes(img.affine)[:3]
        A = img.affine[:3, :3]
        b = img.affine[:3, 3]
        cols = A / sa["voxelsize"]
        sa["xras"] = cols[:, [0]]
        sa["yras"] = cols[:, [1]]
        sa["zras"] = cols[:, [2]]
        sa["cras"] = b + A @ sa["volume"] / 2
        try:
            sa["filename"] = img.file_map["image"].filename
        except Exception:
            pass

        return volgeom

    @classmethod
    def from_string(cls, string):
        """Create a volume structure off of text."""
        volgeom = cls()
        sa = volgeom.structarr
        lines = string.splitlines()
        for key in (
            "valid",
            "filename",
            "volume",
            "voxelsize",
            "xras",
            "yras",
            "zras",
            "cras",
        ):
            label, valstring = lines.pop(0).split(" =")
            assert label.strip() == key

            val = ""
            if valstring.strip():
                parsed = np.genfromtxt(
                    [valstring.encode()], autostrip=True, dtype=cls.dtype[key]
                )
                if parsed.size:
                    val = parsed.reshape(sa[key].shape)
            sa[key] = val
        return volgeom


class FSLinearTransform(LinearTransformStruct):
    """Represents a single LTA's transform structure."""

    template_dtype = np.dtype(
        [
            ("type", "i4"),
            ("mean", "f4", (3, 1)),  # x0, y0, z0
            ("sigma", "f4"),
            ("m_L", "f8", (4, 4)),
            ("m_dL", "f8", (4, 4)),
            ("m_last_dL", "f8", (4, 4)),
            ("src", VolumeGeometry),
            ("dst", VolumeGeometry),
            ("label", "i4"),
        ]
    )
    dtype = template_dtype

    def __getitem__(self, idx):
        """Implement dictionary access."""
        val = super().__getitem__(idx)
        if idx in ("src", "dst"):
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
        current = sa["type"]
        if isinstance(new_type, str):
            new_type = transform_codes.code[new_type]

        if current == new_type:
            return

        # VOX2VOX -> RAS2RAS
        if (current, new_type) == (0, 1):
            src = VolumeGeometry(sa["src"])
            dst = VolumeGeometry(sa["dst"])
            # See https://github.com/freesurfer/freesurfer/
            # blob/bbb2ef78591dec2c1ede3faea47f8dd8a530e92e/utils/transform.cpp#L3696-L3705
            # blob/bbb2ef78591dec2c1ede3faea47f8dd8a530e92e/utils/transform.cpp#L3548-L3568
            M = dst.as_affine() @ sa["m_L"] @ np.linalg.inv(src.as_affine())
            sa["m_L"] = M
            sa["type"] = new_type
            return

        raise NotImplementedError(
            "Converting {} to {} is not yet available".format(
                transform_codes.label[current], transform_codes.label[new_type]
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
        return np.linalg.inv(self.structarr["m_L"])

    def to_string(self, partial=False):
        """Convert this transform to text."""
        sa = self.structarr
        lines = [
            "# LTA file created by NiTransforms",
            "type      = {}".format(sa["type"]),
            "nxforms   = 1",
        ] if not partial else []

        # Standard preamble
        lines += [
            "mean      = {:6.4f} {:6.4f} {:6.4f}".format(*sa["mean"].flatten()),
            "sigma     = {:6.4f}".format(float(sa["sigma"])),
            "1 4 4",
        ]

        # Format parameters matrix
        lines += [
            " ".join(f"{v:18.15e}" for v in sa["m_L"][i])
            for i in range(4)
        ]

        lines += [
            "src volume info",
            str(self["src"]),
            "dst volume info",
            str(self["dst"]),
        ]

        lines += [] if partial else [""]
        return "\n".join(lines)

    @classmethod
    def from_string(cls, string, partial=False):
        """Read a transform from text."""
        lt = cls()
        sa = lt.structarr

        # Drop commented out lines
        lines = _drop_comments(string).splitlines()

        fields = ("type", "nxforms", "mean", "sigma")
        for key in fields[partial * 2:]:
            label, valstring = lines.pop(0).split(" = ")
            assert label.strip() == key

            if key != "nxforms":
                val = np.genfromtxt([valstring.encode()], dtype=cls.dtype[key])
                sa[key] = val.reshape(sa[key].shape)
            else:
                assert valstring.strip() == "1"

        assert lines.pop(0) == "1 4 4"  # xforms, shape + 1, shape + 1
        val = np.genfromtxt([valstring.encode() for valstring in lines[:4]], dtype="f4")
        sa["m_L"] = val
        lines = lines[4:]
        assert lines.pop(0) == "src volume info"
        sa["src"] = np.asanyarray(VolumeGeometry.from_string("\n".join(lines[:8])))
        lines = lines[8:]
        assert lines.pop(0) == "dst volume info"
        sa["dst"] = np.asanyarray(VolumeGeometry.from_string("\n".join(lines)))
        return lt

    @classmethod
    def from_ras(cls, ras, moving=None, reference=None):
        """Create an affine from a nitransform's RAS+ matrix."""
        lt = cls()
        sa = lt.structarr
        sa["sigma"] = 1.0
        sa["mean"] = np.zeros((3, 1), dtype="float")
        sa["type"] = 1  # RAS2RAS
        # Just for reference, nitransforms does not write VOX2VOX
        # PLEASE NOTE THAT LTA USES THE "POINTS" CONVENTION, therefore
        # the source is the reference (coordinates for which we need
        # to find a projection) and destination is the moving image
        # (from which data is pulled-back).
        if reference is not None:
            sa["src"] = np.asanyarray(VolumeGeometry.from_image(reference))

        if moving is not None:
            sa["dst"] = np.asanyarray(VolumeGeometry.from_image(moving))
        # However, the affine needs to be inverted
        # (i.e., it is not a pure "points" convention).
        # This inversion is consistent with self.to_ras()
        sa["m_L"] = np.linalg.inv(ras)
        # to make LTA file format
        return lt


class FSLinearTransformArray(BaseLinearTransformList):
    """A list of linear transforms generated by FreeSurfer."""

    template_dtype = np.dtype(
        [("type", "i4"), ("nxforms", "i4"), ("subject", "U1024"), ("fscale", "f4")]
    )
    dtype = template_dtype
    _inner_type = FSLinearTransform

    def __getitem__(self, idx):
        """Allow dictionary access to the transforms."""
        if idx == "xforms":
            return self._xforms
        if idx == "nxforms":
            return len(self._xforms)
        return self.structarr[idx]

    def to_ras(self, moving=None, reference=None):
        """Set type to RAS2RAS and return the new matrix."""
        self.structarr["type"] = 1
        return [
            xfm.to_ras(moving=moving, reference=reference)
            for xfm in self.xforms
        ]

    def to_string(self):
        """Convert this LTA into text format."""
        code = int(self["type"])
        header = [
            "# LTA-array file created by NiTransforms",
            f"type      = {code} # {transform_codes.label[code]}",
            "nxforms   = {}".format(self["nxforms"]),
        ]
        xforms = [xfm.to_string(partial=True) for xfm in self._xforms]
        footer = [
            "subject {}".format(self["subject"]),
            "fscale {:.6f}".format(float(self["fscale"])),
            "",
        ]
        return "\n".join(header + xforms + footer)

    @classmethod
    def from_string(cls, string):
        """Read this LTA from a text string."""
        lta = cls()
        sa = lta.structarr

        # Drop commented out lines
        lines = _drop_comments(string).splitlines()
        if not lines or not lines[0].startswith("type"):
            raise TransformFileError("Invalid LTA format")

        for key in ("type", "nxforms"):
            label, valstring = lines.pop(0).split(" = ")
            assert label.strip() == key

            val = np.genfromtxt([valstring.encode()], dtype=cls.dtype[key])
            sa[key] = val.reshape(sa[key].shape) if val.size else ""
        for _ in range(sa["nxforms"]):
            lta._xforms.append(
                cls._inner_type.from_string("\n".join(lines[:25]), partial=True)
            )
            lta._xforms[-1].structarr["type"] = sa["type"]
            lines = lines[25:]
        for key in ("subject", "fscale"):
            # Optional keys
            if not (lines and lines[0].startswith(key)):
                continue
            try:
                label, valstring = lines.pop(0).split(" ")
            except ValueError:
                sa[key] = ""
            else:
                assert label.strip() == key

                val = np.genfromtxt([valstring.encode()], dtype=cls.dtype[key])
                sa[key] = val.reshape(sa[key].shape) if val.size else ""

        assert len(lta._xforms) == sa["nxforms"]
        return lta

    @classmethod
    def from_ras(cls, ras, moving=None, reference=None):
        """Create an affine from a nitransform's RAS+ matrix."""
        if ras.ndim == 2:
            return cls._inner_type.from_ras(ras, moving=moving, reference=reference)

        lt = cls()
        sa = lt.structarr
        sa["type"] = 1
        sa["nxforms"] = ras.shape[0]
        for i in range(sa["nxforms"]):
            lt._xforms.append(cls._inner_type.from_ras(
                ras[i, ...], moving=moving, reference=reference
            ))

        sa["subject"] = "unset"
        sa["fscale"] = 0.0
        return lt


def _drop_comments(string):
    """Drop comments."""
    return "\n".join([
        line.split("#")[0].strip()
        for line in string.splitlines()
        if line.split("#")[0].strip()
    ])
