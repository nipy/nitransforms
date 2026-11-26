# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def random_number_generator(request):
    """Automatically set a fixed-seed random number generator for all tests."""
    request.node.rng = np.random.default_rng(1234)
