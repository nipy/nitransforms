# Ubuntu 22.04 LTS - Jammy
ARG BASE_IMAGE=ubuntu:jammy-20240125

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
FROM ${BASE_IMAGE} as downloader
# Bump the date to current to refresh curl/certificates/etc
RUN echo "2023.07.20"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    binutils \
                    bzip2 \
                    ca-certificates \
                    curl \
                    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN update-ca-certificates -f

# FreeSurfer 7.3.2
FROM downloader as freesurfer
COPY docker/files/freesurfer7.3.2-exclude.txt /usr/local/etc/freesurfer7.3.2-exclude.txt
COPY docker/files/fs-cert.pem /usr/local/etc/fs-cert.pem
RUN curl --cacert /usr/local/etc/fs-cert.pem \
     -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/freesurfer-linux-ubuntu22_amd64-7.3.2.tar.gz \
     | tar zxv --no-same-owner -C /opt --exclude-from=/usr/local/etc/freesurfer7.3.2-exclude.txt

# Micromamba
FROM downloader as micromamba

# Install a C compiler to build extensions when needed.
# traits<6.4 wheels are not available for Python 3.11+, but build easily.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /
# Bump the date to current to force update micromamba
RUN echo "2024.02.06"
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

ENV MAMBA_ROOT_PREFIX="/opt/conda"
COPY env.yml /tmp/env.yml
# COPY requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN micromamba create -y -f /tmp/env.yml && \
    micromamba clean -y -a

#
# Main stage
#
FROM ${BASE_IMAGE} as nitransforms

# Configure apt
ENV DEBIAN_FRONTEND="noninteractive" \
    LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8"

# Some baseline tools; bc is needed for FreeSurfer, so don't drop it
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    bc \
                    ca-certificates \
                    curl \
                    libgomp1 \
                    lsb-release \
                    netbase \
                    xvfb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install files from stages
COPY --from=freesurfer /opt/freesurfer /opt/freesurfer
COPY --from=afni/afni_make_build:AFNI_25.2.09 \
    /opt/afni/install/libf2c.so \
    /opt/afni/install/libmri.so \
    /usr/local/lib/
COPY --from=afni/afni_make_build:AFNI_25.2.09 \
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
    FREESURFER_HOME="/opt/freesurfer"
ENV SUBJECTS_DIR="$FREESURFER_HOME/subjects" \
    FUNCTIONALS_DIR="$FREESURFER_HOME/sessions" \
    MNI_DIR="$FREESURFER_HOME/mni" \
    LOCAL_DIR="$FREESURFER_HOME/local" \
    MINC_BIN_DIR="$FREESURFER_HOME/mni/bin" \
    MINC_LIB_DIR="$FREESURFER_HOME/mni/lib" \
    MNI_DATAPATH="$FREESURFER_HOME/mni/data"
ENV PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    MNI_PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    PATH="$FREESURFER_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH"

# AFNI config
ENV PATH="/opt/afni-latest:$PATH" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_PLUGINPATH="/opt/afni-latest"

# Workbench config
ENV PATH="/opt/workbench/bin_linux64:$PATH"

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users neuro
WORKDIR /home/neuro
ENV HOME="/home/neuro" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=micromamba /opt/conda/envs/nitransforms /opt/conda/envs/nitransforms

ENV MAMBA_ROOT_PREFIX="/opt/conda"
RUN micromamba shell init -s bash && \
    echo "micromamba activate nitransforms" >> $HOME/.bashrc
ENV PATH="/opt/conda/envs/nitransforms/bin:$PATH" \
    CPATH="/opt/conda/envs/nitransforms/include:$CPATH" \
    LD_LIBRARY_PATH="/opt/conda/envs/nitransforms/lib:$LD_LIBRARY_PATH"

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

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Install package
# CRITICAL: Make sure python setup.py --version has been run at least once
#           outside the container, with access to the git history.
COPY --from=src /src/dist/*.whl .
RUN python -m pip install --no-cache-dir $( ls *.whl )[all]


RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} + && \
    rm -rf $HOME/.npm $HOME/.conda $HOME/.empty

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
