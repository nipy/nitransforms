from argparse import ArgumentParser, RawDescriptionHelpFormatter
from textwrap import dedent

import nibabel as nb

from .linear import load as linload
from .nonlinear import load as nlinload

def cli_apply(**kwargs):
    """
    Apply a transformation to an image, resampling on the reference

    Usage:

        $ nt apply moving.nii.gz xform.fsl --reference reference.nii.gz
    """
    try:
        xfm = linload(xform)
    except:
        xfm = nlinload(xform)

    xfm.apply(moving, **kwargs)


def get_parser():
    desc = dedent(
        """
        NiTransforms command-line utility.

        Commands:

            apply       Apply a transformation to an image

        For command specific information, use 'nt <command> -h'.
        """
    )

    parser = ArgumentParser(
        description=desc, formatter_class=RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command')
    def _add_subparser(name, description):
        subp = subparsers.add_parser(
            name,
            description=dedent(description),
            formatter_class=RawDescriptionHelpFormatter,
        )
        return subp

    applyp = _add_subparser('apply', cli_apply.__doc__)
    applyp.add_argument('transform', help='The transform file')
    applyp.add_argument(
        'moving', help='The image containing the data to be resampled'
    )
    applyp.add_argument('reference', help='The reference space to resample onto')
    applyp.add_argument('--order', help='The order of the spline transformation')
    applyp.add_argument(
        '--mode', 
        help='Determines how the input image is extended when the resampling overflows a border'
    )
    applyp.add_argument('--cval', help='Constant used when using "constant" mode')
    applyp.add_argument(
        '--prefilter', 
        action='store_true', 
        help="Determines if the image's data array is prefiltered with a spline filter before "
             "interpolation"
    )
    return parser


def main(pargs=None):
    parser = get_parser()
    pargs = parser.parse_args(pargs)
    if pargs.command is None:
        parser.print_help()
        return

    if pargs.command == 'apply':
        cli_apply(**vars(pargs))
