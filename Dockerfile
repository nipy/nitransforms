# Ubuntu 22.04 LTS - Jammy
ARG BASE_IMAGE=ubuntu:jammy-20250730

#
# Build wheel
#
FROM ghcr.io/astral-sh/uv:python3.13-alpine AS src
RUN apk add git
COPY . /src
RUN uv build --wheel /src

#
# Download stages
#

# Utilities for downloading packages
FROM ${BASE_IMAGE} AS downloader
# Bump the date to current to refresh curl/certificates/etc
RUN echo "2025.09.25"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    binutils \
                    bzip2 \
                    ca-certificates \
                    curl \
                    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN update-ca-certificates -f

# Micromamba
FROM downloader AS micromamba

WORKDIR /
# Bump the date to current to force update micromamba
RUN echo "2025.09.05"
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

ENV MAMBA_ROOT_PREFIX="/opt/conda"
COPY env.yml /tmp/env.yml
WORKDIR /tmp
RUN micromamba create -y -f /tmp/env.yml && \
    micromamba clean -y -a

#
# Main stage
#
FROM ${BASE_IMAGE} AS nitransforms

# Configure apt
ENV DEBIAN_FRONTEND="noninteractive" \
    LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libexpat1 \
        libgomp1 \
        && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install FreeSurfer and AFNI bins from images
COPY --from=freesurfer/freesurfer:7.4.1 \
    /usr/local/freesurfer/bin/mri_vol2vol \
    /usr/local/freesurfer/bin/
COPY --from=afni/afni_make_build:AFNI_25.2.09 \
    /opt/afni/install/libf2c.so \
    /opt/afni/install/libmri.so \
    /usr/local/lib/
COPY --from=afni/afni_make_build:AFNI_25.2.09 \
    /opt/afni/install/3dAllineate \
    /opt/afni/install/3dNwarpApply \
    /opt/afni/install/3dWarp \
    /opt/afni/install/3drefit \
    /opt/afni/install/3dvolreg \
    /usr/local/bin/

# Simulate SetUpFreeSurfer.sh
ENV OS="Linux" \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" \
    FREESURFER_HOME="/usr/local/freesurfer"
ENV SUBJECTS_DIR="$FREESURFER_HOME/subjects" \
    FUNCTIONALS_DIR="$FREESURFER_HOME/sessions" \
    LOCAL_DIR="$FREESURFER_HOME/local" \
    PATH="$FREESURFER_HOME/bin:$PATH"

# AFNI config
ENV AFNI_IMSAVE_WARNINGS="NO"

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users neuro
WORKDIR /home/neuro
ENV HOME="/home/neuro"

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=micromamba /opt/conda/envs/nitransforms /opt/conda/envs/nitransforms

ENV MAMBA_ROOT_PREFIX="/opt/conda"
RUN micromamba shell init -s bash && \
    echo "micromamba activate nitransforms" >> $HOME/.bashrc
ENV PATH="/opt/conda/envs/nitransforms/bin:$PATH"

# FSL environment
ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1 \
    FSLDIR="/opt/conda/envs/nitransforms" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q"

# Install package
COPY --from=src /src/dist/*.whl .
RUN python -m pip install --no-cache-dir $( ls *.whl )[all]

RUN ldconfig
WORKDIR /tmp/

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="nitransforms" \
      org.label-schema.vcs-url="https://github.com/nipy/nitransforms" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
