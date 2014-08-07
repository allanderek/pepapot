#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()

setup(
    name='pepapot',
    version='0.1.0',
    description="""An attempt at writing a very simple PEPA tool in Python. 
                   It is intended to be a compliment to pyPEPA. Here though 
                   the focus is on being as simple as possible and hence can 
                   be used in, for example, student projects. This is now also
                   an openshift web app.
                """,
    long_description=readme + '\n\n',
    author='Allan Clark',
    author_email='allan.clark@gmail.com',
    url='https://github.com/allanderek/pepapot',
    packages=[
        'pepapot',
    ],
    package_dir={'pepapot': 'pepapot'},
    include_package_data=True,
    install_requires=["pyparsing", "numpy", "lazy", "scipy", "docopt",
                      "bottle", "jinja2", "SQLAlchemy",
                      "mysql-connector-python"],
    license="BSD",
    zip_safe=False,
    keywords='pepapot',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
    test_suite='tests',
)
