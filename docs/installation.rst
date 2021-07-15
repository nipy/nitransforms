Installation
============
*NiTransforms* is distributed via *Pypi* and can easily be installed
within your Python distribution with::

  python -m pip install nitransforms

Alternatively, you can install the bleeding-edge version of the software
directly from the GitHub repo with::

  python -m pip install git+https://github.com/poldracklab/nitransforms.git@master

To verify the installation, you can run the following command::

  python -c "import nitransforms as nt; print(nt.__version__)"

You should see the version number.

Developers
----------
Advanced users and developers who plan to contribute with bugfixes, documentation,
etc. can first clone our Git repository::

  git clone https://github.com/poldracklab/nitransforms.git


and install the tool in *editable* mode::

  cd nitransforms
  python -m pip install -e .
