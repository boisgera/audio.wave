#!/usr/bin/env python

# Python 2.7 Standard Library
import sys

# Pip Package Manager
try:
    import pip
    import setuptools
    import pkg_resources
except ImportError:
    error = "pip is not installed, refer to <{url}> for instructions."
    raise ImportError(error.format(url="http://pip.readthedocs.org"))

# Numpy
import numpy

# Extra Third-Party Libraries
sys.path.insert(1, ".lib")
try:
    setup_requires = ["about>=4.0.0"]
    require = lambda *r: pkg_resources.WorkingSet().require(*r)
    require(*setup_requires)
    import about
except pkg_resources.DistributionNotFound:
    error = """{req!r} not found; install it locally with:

    pip install --target=.lib --ignore-installed {req!r}
"""
    raise ImportError(error.format(req=" ".join(setup_requires)))
import about

# Project Metadata 
sys.path.insert(1, "audio")
import about_wave


info = dict(
  metadata     = about.get_metadata(about_wave),
  code         = dict(packages=setuptools.find_packages()),
  data         = {},
  requirements = dict(install_requires="bitstream logfile wish".split()),
  scripts      = {},
  commands     = {},
  plugins      = {},
  tests        = dict(test_suite="test.suite"),
)

if __name__ == "__main__":
    kwargs = {k:v for dct in info.values() for (k,v) in dct.items()}
    setuptools.setup(**kwargs)

