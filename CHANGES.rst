25.0.1 (August 14, 2025)
========================
A patch release addressing a critical issue in the ``ImageGrid`` class relating coordinates,
and bolstered the testing of ANTs' generated displacements fields by cross-comparing against
``antsApplyTransformsToPoints`` in several new tests.

CHANGES
-------
* FIX: ``ImageGrid._coords`` was somehow overwritten + re-enable tests by @oesteban in https://github.com/nipy/nitransforms/pull/276
* FIX: Revision of index and RAS coordinate grids generation by @oesteban in https://github.com/nipy/nitransforms/pull/271
* ENH: Implement ITK densefields checks vs ``antsApplyTransformsToPoints`` by @oesteban in https://github.com/nipy/nitransforms/pull/277
* ENH: Add unit test on dense fields (extracted from #266) by @oesteban in https://github.com/nipy/nitransforms/pull/274
* RF: Move tests to better locations by @oesteban in https://github.com/nipy/nitransforms/pull/272
* MNT: Minimal housekeeping of tests by @oesteban in https://github.com/nipy/nitransforms/pull/275
* MNT: Fix coverage XML path in CircleCI by @oesteban in https://github.com/nipy/nitransforms/pull/265
* MNT: Add test cases demonstrating ordering bug reading composite ITK's HDF5 files by @oesteban in https://github.com/nipy/nitransforms/pull/263
* STY: Run ruff at the source root by @oesteban in https://github.com/nipy/nitransforms/pull/273

**Full Changelog**: https://github.com/nipy/nitransforms/compare/25.0.0...25.0.1

25.0.0 (July 22, 2025)
======================
A new major release introducing critical fixes and important new functionality.
Most notably, it includes a hotfix for 4D resampling (also backported to 24.1.4) and adds **experimental support for the X5 format** (*BIDS*).
The X5 support enables I/O for linear and nonlinear transforms and includes partial support for transformation chains—marking a first step
toward full *BIDS* interoperability.
This release also drops support for *Python* 3.9 and earlier, aligning with modern *Python* standards and paving the way for future improvements.

CHANGES
-------
* FIX: BSpline mapping of individual points by @oesteban in https://github.com/nipy/nitransforms/pull/256
* FIX: Remove implementation of an abstract class by @oesteban in https://github.com/nipy/nitransforms/pull/255
* FIX: Add test for ``DenseFieldTransform`` handling of OOB points by @oesteban in https://github.com/nipy/nitransforms/pull/254
* ENH: X5 read/write support of ``TransformChain`` by @oesteban in https://github.com/nipy/nitransforms/pull/253
* ENH: Loading of X5 (linear) transforms by @oesteban in https://github.com/nipy/nitransforms/pull/243
* ENH: Implement X5 representation and output to filesystem by @oesteban in https://github.com/nipy/nitransforms/pull/241
* DOC: Fix references to ``os.PathLike`` by @oesteban in https://github.com/nipy/nitransforms/pull/242
* MNT: Drop Python 3.9 by @oesteban in https://github.com/nipy/nitransforms/pull/259
* MNT: Increase coverage by testing edge cases and adding docstrings by @oesteban in https://github.com/nipy/nitransforms/pull/248
* MNT: Refactor io/lta to reduce one partial line by @oesteban in https://github.com/nipy/nitransforms/pull/246
* MNT: Move flake8 config into ``pyproject.toml`` by @oesteban in https://github.com/nipy/nitransforms/pull/245
* MNT: Configure coverage to omit tests by @oesteban in https://github.com/nipy/nitransforms/pull/244

**Full Changelog**: https://github.com/nipy/nitransforms/compare/24.1.4...25.0.0

24.1.4 (July 20, 2025)
======================
Hotfix release addressing an issue in dense displacements fields.

CHANGES
-------
* FIX: Backport https://github.com/nipy/nitransforms/pull/251 into 24.1.x series by @oesteban in https://github.com/nipy/nitransforms/pull/252

24.1.3 (July 19, 2025)
======================
Hotfix release addressing the issues when resampling 4D data.

CHANGES
-------
* FIX: Broken 4D resampling by @oesteban in https://github.com/nipy/nitransforms/pull/247

24.1.2 (June 02, 2025)
======================
New patch release that addresses a crash when applying a 3D transform to a 4D image.

New Contributors
----------------
* @coryshain made their first contribution in https://github.com/nipy/nitransforms/pull/236

CHANGES
-------
* FIX: Patch for crash when applying 3D transform to 4D image (#236)
* MNT: Switch from zenodo.json to CITATION.cff, add contributors (#237)

**Full Changelog**: https://github.com/nipy/nitransforms/compare/24.1.1...24.1.2

24.1.1 (December 18, 2024)
==========================
New patch release that adds ``nitransforms.resampling.apply`` as a top-level import, and removes the `pkg_resources` dependency.

CHANGES
-------

* RF: Add nitransforms.resamping.apply to top module imports in https://github.com/nipy/nitransforms/pull/227
* FIX: Remove pkg_resources dependency in https://github.com/nipy/nitransforms/pull/230

**Full Changelog**: https://github.com/nipy/nitransforms/compare/24.1.0...24.1.1

24.1.0 (November 17, 2024)
==========================
New feature release in the 24.1.x series.

This release has the same code as 24.0.2, but the package has been
tested with Numpy 2.0 and Python 3.13 and the metadata updated accordingly.

CHANGES
-------
* MAINT: Transition to pyproject.toml and tox, support numpy 2, python 3.13
  by @effigies in https://github.com/nipy/nitransforms/pull/228

**Full Changelog**: https://github.com/nipy/nitransforms/compare/24.0.2...24.1.0

24.0.2 (September 21, 2024)
===========================
Bug-fix release in the 24.0.x series.

CHANGES
-------

* FIX: Add per-volume transforms as single transform in chain by @effigies in https://github.com/nipy/nitransforms/pull/226

**Full Changelog**: https://github.com/nipy/nitransforms/compare/24.0.1...24.0.2

24.0.1 (September 17, 2024)
===========================
Bug-fix release in the 24.0.x series.

New Contributors
----------------
* @shnizzedy made their first contribution in https://github.com/nipy/nitransforms/pull/222

CHANGES
-------

* FIX: Use standard library ``pathlib`` by @shnizzedy in https://github.com/nipy/nitransforms/pull/222
* MAINT: Support pre-``__or__`` types by @effigies in https://github.com/nipy/nitransforms/pull/223
* MAINT: Bump the actions-infrastructure group with 3 updates by @dependabot in https://github.com/nipy/nitransforms/pull/224
* MAINT: Bump codecov/codecov-action from 3 to 4 by @dependabot in https://github.com/nipy/nitransforms/pull/225

**Full Changelog**: https://github.com/nipy/nitransforms/compare/24.0.0...24.0.1

24.0.0 (August 18, 2024)
========================
A new series incorporating several major changes, including bugfixes and taking on several
housekeeping/maintenance actions.
One relevant change is the outsourcing of the ``apply()`` member out of
transformation data structures by @jmarabotto.
The method ``apply()`` is now a standalone method that operates on one transform
and images/surfaces/etc. provided as arguments.
A later major development is the adoption of a foundation for surface transforms by @feilong
and @Shotgunosine.

New Contributors
----------------

* @mvdoc made their first contribution in https://github.com/nipy/nitransforms/pull/194
* @jmarabotto made their first contribution in https://github.com/nipy/nitransforms/pull/197
* @bpinsard made their first contribution in https://github.com/nipy/nitransforms/pull/182
* @jbanusco made their first contribution in https://github.com/nipy/nitransforms/pull/188
* @feilong made their first contribution in https://github.com/nipy/nitransforms/pull/203

CHANGES
-------

* FIX: Inefficient iterative reloading of reference and moving images by @oesteban in https://github.com/nipy/nitransforms/pull/186
* FIX: Postpone coordinate mapping on linear array transforms by @oesteban in https://github.com/nipy/nitransforms/pull/187
* FIX: Remove unsafe cast during ``TransformBase.apply()`` by @effigies in https://github.com/nipy/nitransforms/pull/189
* FIX: ``_is_oblique()`` by @mvdoc in https://github.com/nipy/nitransforms/pull/194
* FIX: Update implementation of ``ndim`` property of transforms by @jmarabotto in https://github.com/nipy/nitransforms/pull/197
* FIX: Output displacement fields by @bpinsard in https://github.com/nipy/nitransforms/pull/182
* FIX: Composition of deformation fields by @jbanusco in https://github.com/nipy/nitransforms/pull/188
* FIX: Indexing disallowed in lists introduced by bugfix by @oesteban in https://github.com/nipy/nitransforms/pull/204
* FIX: Do not transpose (see :obj:`~scipy.ndimage.map_coordinates`) by @oesteban in https://github.com/nipy/nitransforms/pull/207
* FIX: Forgotten test using ``xfm.apply()`` by @oesteban in https://github.com/nipy/nitransforms/pull/208
* FIX: Load ITK fields from H5 correctly by @effigies in https://github.com/nipy/nitransforms/pull/211
* FIX: Wrong warning argument name ``level`` in ``warnings.warn`` by @oesteban in https://github.com/nipy/nitransforms/pull/216
* ENH: Define ``ndim`` property on nonlinear transforms by @oesteban in https://github.com/nipy/nitransforms/pull/201
* ENH: Outsource ``apply()`` from transform objects by @jmarabotto in https://github.com/nipy/nitransforms/pull/195
* ENH: Restore ``apply()`` method, warning of deprecation and calling function by @effigies in https://github.com/nipy/nitransforms/pull/209
* ENH: ``SurfaceTransform`` class by @feilong in https://github.com/nipy/nitransforms/pull/203
* ENH: reenable-parallelization-apply-214 (builds on PR #215, solves Issue #214) by @jmarabotto in https://github.com/nipy/nitransforms/pull/217
* ENH: Parallelize serialized 3D+t transforms by @oesteban in https://github.com/nipy/nitransforms/pull/220
* ENH: Implement a memory limitation mechanism in loading data by @oesteban in https://github.com/nipy/nitransforms/pull/221
* ENH: Serialize+parallelize 4D ``apply()`` into 3D+t and add 'low memory' loading by @oesteban in https://github.com/nipy/nitransforms/pull/215
* MAINT: Loosen dependencies by @mgxd in https://github.com/nipy/nitransforms/pull/164
* MAINT: Drop Python 3.7 support, test through 3.11 by @effigies in https://github.com/nipy/nitransforms/pull/181
* MAINT: Update CircleCI's infrastructure (machine image and Python version in Docker image) by @oesteban in https://github.com/nipy/nitransforms/pull/206
* MAINT: Fix tests for Python 3.12, numpy 2.0, and pytest-xdist by @effigies in https://github.com/nipy/nitransforms/pull/210
* MAINT: Update ANTs' pinnings by @oesteban in https://github.com/nipy/nitransforms/pull/219

**Full Changelog**: https://github.com/nipy/nitransforms/compare/23.0.1...24.0.0

23.0.1 (July 10, 2023)
======================
Hotfix release addressing two issues.

CHANGES
-------

* FIX: Load ITK's ``.mat`` files with ``Affine``'s loaders (#179)
* FIX: numpy deprecation errors after 1.22 (#180)


23.0.0 (June 13, 2023)
======================
A new major release preparing for the finalization of the package and migration into
NiBabel, mostly addressing bugfixes and scheduled added new features.

CHANGES
-------

* FIX: Set x-forms on resampled images (#176)
* FIX: Ensure datatype of generated CIFTI2 file in ``TransformBase`` unit test (#178)
* ENH: Read ITK's composite transforms with only affines (#174)
* ENH: "Densify" voxel-wise nonlinear mappings with interpolation  (#168)
* ENH: Extend the nonlinear transforms API (#166)
* ENH: API change in ``TransformChain`` - new composition convention (#165)
* MAINT: Rotate CircleCI secrets and setup up org-level context (#172)

22.0.1 (April 28, 2022)
=======================
A patch release after migration into the NiPy organization.
This release is aliased as 21.0.1 to flexibilize dependency resolution.

CHANGES
-------

* FIX: Orientation of displacements field and header when reading ITK's h5 (#162)
* FIX: Wrong datatype used for offset when reading ITK's h5 fields. (#161)
* ENH: Guess open linear transform formats (#160)
* MAINT: Conclude migration ``poldracklab`` -> ``nipy`` (#163)

22.0.0 (February 28, 2022)
==========================
The first stable release of *NiTransforms* in 2022.
Contains all the new bug-fixes, features, and maintenance executed within the
context of the NiBabel EOSS4 grant from the CZI Foundation.

CHANGES
-------

* FIX: Implement AFNI's deoblique operations (#117)
* FIX: Ensure input dtype is kept after resampling (#153)
* FIX: Replace deprecated ``_read_mat`` with ``scipy.io.loadmat`` (#151)
* FIX: Add FSL-LTA-FSL regression tests (#146)
* FIX: Increase FSL serialization precision (#144)
* FIX: Refactor of LTA implementation (#145)
* FIX: Load arrays of linear transforms from AFNI files (#143)
* FIX: Load arrays of linear transforms from FSL files (#142)
* FIX: Double-check dtypes within tests and increase RMSE tolerance (#141)
* ENH: Base implementation of B-Spline transforms (#138)
* ENH: I/O of FSL displacements fields (#51)
* MAINT: Fix path to test summaries in CircleCI (#148)
* MAINT: Move testdata on to gin.g-node.org & datalad (#140)
* MAINT: scipy-1.8, numpy-1.22 require python 3.8 (#139)

21.0.0 (September 10, 2021)
===========================
A first release of *NiTransforms*.
This release accompanies a corresponding `JOSS submission <https://doi.org/10.21105/joss.03459>`__.

CHANGES
-------

* FIX: Final edits to JOSS submission (#135)
* FIX: Add mention to potential alternatives in JOSS submission (#132)
* FIX: Misinterpretation of voxel ordering in LTAs (#129)
* FIX: Suggested edits to the JOSS submission (#121)
* FIX: Invalid DOI (#124)
* FIX: Remove the ``--inv`` flag from regression ``mri_vol2vol`` regression test (#78)
* FIX: Improve handling of optional fields in LTA (#65)
* FIX: LTA conversions (#36)
* ENH: Add more comprehensive comments to notebook (#134)
* ENH: Add an ``.asaffine()`` member to ``TransformChain`` (#90)
* ENH: Read (and apply) *ITK*/*ANTs*' composite HDF5 transforms (#79)
* ENH: Improved testing of LTA handling - *ITK*-to-LTA, ``mri_concatenate_lta`` (#75)
* ENH: Add *FS* transform regression (#74)
* ENH: Add *ITK*-LTA conversion test (#66)
* ENH: Support for transforms mappings (e.g., head-motion correction) (#59)
* ENH: command line interface (#55)
* ENH: Facilitate loading of displacements field transforms (#54)
* ENH: First implementation of *AFNI* displacement fields (#50)
* ENH: Base implementation of transforms chains (composition) (#43)
* ENH: First implementation of loading and applying *ITK* displacements fields (#42)
* ENH: Refactor of *AFNI* and *FSL* I/O with ``StringStructs`` (#39)
* ENH: More comprehensive implementation of ITK affines I/O (#35)
* ENH: Added some minimal test-cases to the Affine class (#33)
* ENH: Rewrite load/save utilities for ITK's MatrixOffsetBased transforms in ``io`` (#31)
* ENH: Rename ``resample()`` with ``apply()`` (#30)
* ENH: Write tests pulling up the coverage of base submodule (#28)
* ENH: Add tests and implementation for Displacements fields and refactor linear accordingly (#27)
* ENH: Uber-refactor of code style, method names, etc. (#24)
* ENH: Increase coverage of linear transforms code (#23)
* ENH: FreeSurfer LTA file support (#17)
* ENH: Use ``obliquity`` directly from nibabel (#18)
* ENH: Setting up a battery of tests (#9)
* ENH: Revise doctests and get them ready for more thorough testing. (#10)
* DOC: Add *Zenodo* metadata record (#136)
* DOC: Better document the *IPython* notebooks (#133)
* DOC: Transfer ``CoC`` from *NiBabel* (#131)
* DOC: Clarify integration plans with *NiBabel* in the ``README`` (#128)
* DOC: Add contributing page to RTD (#130)
* DOC: Add ``CONTRIBUTING.md`` file pointing at *NiBabel* (#127)
* DOC: Add example notebooks to sphinx documentation (#126)
* DOC: Add an *Installation* section (#122)
* DOC: Display API per module (#120)
* DOC: Add figure to JOSS draft / Add @smoia to author list (#61)
* DOC: Initial JOSS draft (#47)
* MAINT: Add imports of modules in ``__init__.py`` to workaround #91 (#92)
* MAINT: Fix missing ``python3`` binary on CircleCI build job step (#85)
* MAINT: Use ``setuptools_scm`` to manage versioning (#83)
* MAINT: Split binary test-data out from gh repo (#84)
* MAINT: Add Docker image/circle build (#80)
* MAINT: Drop Python 3.5 (#77)
* MAINT: Better config on ``setup.py`` (binary operator starting line) (#60)
* MAINT: add docker build to travis matrix (#29)
* MAINT: testing coverage (#16)
* MAINT: pep8 complaints (#14)
* MAINT: skip unfinished implementation tests (#15)
* MAINT: pep8speaks (#13)
