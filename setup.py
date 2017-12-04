#!/usr/bin/env python
######################################################################
# Copyright (c) 2015, Max Planck Society
# \file setup.py
# \author Franziska Meier
# \author Daniel Kappler
#######################################################################
from setuptools import setup

__author__ = 'Franziska Meier, Daniel Kappler'
__copyright__ = '2015, Max Planck Society'


setup(
    name='meta_learning',
    author='Franziska Meier, Daniel Kappler',
    author_email='franzi.meier@gmail.com, daniel.kappler@gmail.com',
    version=1.0,
    packages=['meta_learning'],
    package_dir={'meta_learning': ''},
    install_requires=[
        'pyyaml',
        'ipdb',
        'jinja2',
        'easydict',
        'jupyter',
        'h5py',
        'jinja2',
        'progressbar2',
        'dill'
    ],
    zip_safe=False
)
