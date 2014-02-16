#!/usr/bin/env python

# Python 2.7 Standard Library
pass

# Third-Party Library
import about
import setuptools

metadata = about.get_metadata("about_wave", "audio")
contents = dict(packages=setuptools.find_packages())
requirements = dict(install_requires=\
                    ["numpy", "bitstream", "script", "lsprofcalltree"])

info = {}
info.update(metadata)
info.update(contents)
info.update(requirements)

if __name__ == "__main__":
    setuptools.setup(**info)

