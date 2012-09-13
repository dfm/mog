#!/usr/bin/env python

try:
    from setuptools import setup, Extension
    setup, Extension
except ImportError:
    print "failed import"
    from distutils.core import setup
    from distutils.extension import setup, Extension
    setup, Extension

import numpy.distutils.misc_util


algorithms_ext = Extension("mog._algorithms", ["mog/_algorithms.c"])

setup(
    name="mog",
    version="0.0.1",
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    packages=["mog"],
    ext_modules = [algorithms_ext],
    description="Mixtures of Gaussians.",
    long_description=open("README.rst").read(),
    install_requires=["numpy"],
    include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs(),
)
