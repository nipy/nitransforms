---
title: 'NiTransforms: A Python tool to read, represent, manipulate, and apply $n$-dimensional spatial transforms'
tags:
  - Python
  - neuroimaging
  - image processing
  - spatial transform
  - nibabel
authors:
  - name: Mathias Goncalves
    orcid: 0000-0002-7252-7771
    affiliation: 1
  - name: Christopher J. Markiewicz
    orcid: 0000-0002-6533-164X
    affiliation: "1, 2"
  - name: Stefano Moia
    orcid: 0000-0002-2553-3327
    affiliation: "4"
  - name: Satrajit S. Ghosh
    orcid: 0000-0002-5312-6729
    affiliation: "2, 3"
  - name: Russell A. Poldrack
    orcid: 0000-0001-6755-0259
    affiliation: 1
  - name: Oscar Esteban
    orcid: 0000-0001-8435-6191
    affiliation: 1
affiliations:
 - name: Department of Psychology, Stanford University, Stanford, CA, USA
   index: 1
 - name: McGovern Institute for Brain Research, Massachusetts Institute of Technology (MIT), Cambridge, MA, USA
   index: 2
 - name: Department of Otolaryngology, Harvard Medical School, Boston, MA, USA
   index: 3
 - name: Basque Center on Cognition Brain and Language, San Sebastian, Spain
   index: 4
date: 04 November 2019
bibliography: nt.bib
---

# Summary

Spatial transforms formalize mappings between coordinates of objects in biomedical images.
Transforms typically are the outcome of image registration methodologies, which estimate the alignment between two images.
Image registration is a prominent task present in image processing.
In neuroimaging, the proliferation of image registration software implementations has resulted in a disparate collection of structures and file formats used to preserve and communicate the transformation.
This assortment of formats presents the challenge of compatibility between tools and endangers the reproducibility of results.

_NiTransforms_ is a Python tool capable of reading and writing tranforms produced by the most popular neuroimaging software (AFNI [@cox_software_1997], FSL [@jenkinson_fsl_2012], FreeSurfer [@fischl_freesurfer_2012], ITK via ANTs [@avants_symmetric_2008], and SPM [@friston_statistical_2006]).
Additionally, the tool provides seamless conversion between these formats, as well as the ability of applying the transforms to other images.
_NiTransforms_ is inspired by `NiBabel` [@brett_nibabel_2006], a Python package with a collection of tools to read, write and handle neuroimaging data, and will be included as a new module.

# Spatial transforms

Let $\vec{x}$ represent the coordinates of a point in the reference coordinate system $R$, and $\vec{x}'$ its projection on to another coordinate system $M$:

$T\colon R \subset \mathbb{R}^n \to M \subset \mathbb{R}^n$

$\vec{x} \mapsto \vec{x}' = f(\vec{x}).$

In an image registration problem, $M$ is a moving image from which we want to sample data in order to bring the image into spatial alignment with the reference image $R$.
Hence, $f$ here is the spatial transformation function that maps from coordinates in $R$ to coordinates in $M$.
There are a multiplicity of image registration algorithms and corresponding image transformation models to estimate linear and nonlinear transforms.

The problem has been traditionally confused by the need of _transforming_ or mapping one image (generally referred to as _moving_) into another that serves as reference, with the goal of _fusing_ the information from both.
An example of image fusion application would be the alignment of functional data from one individual's brain to the same individual's corresponding anatomical MRI scan for visualization.
Therefore, "applying a transform" entails two operations: first, transforming the coordinates of the samples in the reference image $R$ to find their mapping $\vec{x}'$ on $M$ via $T\{\cdot\}$, and second an interpolation step as $\vec{x}'$ will likely fall off-the-grid of the moving image $M$.
These two operations are confusing because, while the spatial transformation projects from $R$ to $M$, the data flows in reversed way after the interpolation of the values of $M$ at the mapped coordinates $\vec{x}'$.

![figure1](https://github.com/poldracklab/nitransforms/raw/master/docs/_static/figure1-joss.png "Figure 1")

# Software Architecture

There are four main components within the tool: an `io` submodule to handle the structure of the various file formats, a `base` submodule where abstract classes are defined, a `linear` submodule implementing $n$-dimensional linear transforms, and a `nonlinear` submodule for both parametric and non-parametric nonlinear transforms.
Furthermore, _NiTranforms_ provides a straightforward _Application Programming Interface_ (API) that allows researchers to map point sets via transforms, as well as apply transforms (i.e., mapping the coordinates and interpolating the data) to data structures with ease.

To ensure the consistency and uniformity of internal operations, all transforms are defined using a left-handed coordinate system of physical coordinates.
In words from the neuroimaging domain, the coordinate system of transforms is _RAS+_ (or positive directions point to the Righthand for the first axis, Anterior for the second, and Superior for the third axis).
The internal representation of transform coordinates is the most relevant design decision, and implies that a conversion of coordinate system is necessary to correctly interpret transforms generated by other software.
When a transform that is defined in another coordinate system is loaded, it is automatically converted into _RAS+_ space.

_NiTransforms_ was developed using a test-driven development paradigm, with the
battery of tests being written prior to the software implementations.
Two categories of tests were used: unit tests and cross-tool comparison tests.
Unit tests evaluate the formal correctness of the implementation, while cross-tool
comparison tests assess the correct implementation of third-party software.
The testing suite is incorporated into a continuous integration framework, which assesses the continuity of the implementation along the development life and ensures that code changes and additions do not break existing functionalities.

# References
