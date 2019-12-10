from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os
from textwrap import dedent


from .linear import load as linload
from .nonlinear import load as nlinload


def cli_apply(pargs):
    """
    Apply a transformation to an image, resampling on the reference.

    Sample usage:

        $ nt apply xform.fsl moving.nii.gz --ref reference.nii.gz --out moved.nii.gz

        $ nt apply warp.nii.gz moving.nii.gz --fmt afni --nonlinear

    """
    fmt = pargs.fmt or pargs.transform.split('.')[-1]
    if fmt in ('tfm', 'mat', 'h5', 'x5'):
        fmt = 'itk'
    elif fmt == 'lta':
        fmt = 'fs'

    if fmt not in ('fs', 'itk', 'fsl', 'afni', 'x5'):
        raise ValueError(
            "Cannot determine transformation format, manually set format with the `--fmt` flag"
        )

    if pargs.nonlinear:
        xfm = nlinload(pargs.transform, fmt=fmt)
    else:
        xfm = linload(pargs.transform, fmt=fmt)

    # ensure a reference is set
    xfm.reference = pargs.ref or pargs.moving

    moved = xfm.apply(
        pargs.moving,
        order=pargs.order,
        mode=pargs.mode,
        cval=pargs.cval,
        prefilter=pargs.prefilter
    )
    moved.to_filename(
        pargs.out or "nt_{}".format(os.path.basename(pargs.moving))
    )


def get_parser():
    desc = dedent("""
        NiTransforms command-line utility.

        Commands:

            apply       Apply a transformation to an image

        For command specific information, use 'nt <command> -h'.
    """)

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
    applyp.set_defaults(func=cli_apply)
    applyp.add_argument('transform', help='The transform file')
    applyp.add_argument(
        'moving', help='The image containing the data to be resampled'
    )
    applyp.add_argument('--ref', help='The reference space to resample onto')
    applyp.add_argument(
        '--fmt',
        choices=('itk', 'fsl', 'afni', 'fs', 'x5'),
        help='Format of transformation. If no option is passed, nitransforms will '
        'estimate based on the transformation file extension.'
    )
    applyp.add_argument(
        '--out', help="The transformed image. If not set, will be set to `nt_{moving}`"
    )
    applyp.add_argument(
        '--nonlinear', action='store_true', help='Transformation is nonlinear (default: False)'
    )
    applykwargs = applyp.add_argument_group('Apply customization')
    applykwargs.add_argument(
        '--order',
        type=int,
        default=3,
        choices=range(6),
        help='The order of the spline transformation (default: 3)'
    )
    applykwargs.add_argument(
        '--mode',
        choices=('constant', 'reflect', 'nearest', 'mirror', 'wrap'),
        default='constant',
        help='Determines how the input image is extended when the resampling overflows a border '
        '(default: constant)'
    )
    applykwargs.add_argument(
        '--cval',
        type=float,
        default=0.0,
        help='Constant used when using "constant" mode (default: 0.0)'
    )
    applykwargs.add_argument(
        '--prefilter',
        action='store_false',
        help="Determines if the image's data array is prefiltered with a spline filter before "
        "interpolation (default: True)"
    )
    return parser, subparsers


def main(pargs=None):
    parser, subparsers = get_parser()
    pargs = parser.parse_args(pargs)

    try:
        pargs.func(pargs)
    except Exception as e:
        subparser = subparsers.choices[pargs.command]
        subparser.print_help()
        raise(e)
