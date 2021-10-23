#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Miaoqi Chu",
    author_email='mqichu@anl.gov',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A GUI tool to create mask and qmap for xarea detectors used in xray experiments",
    entry_points={
        'console_scripts': [
            'pysimplemask=pysimplemask.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pysimplemask',
    name='pysimplemask',
    packages=find_packages(include=['pysimplemask', 'pysimplemask.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/AZjk/pysimplemask',
    version='0.0.1',
    zip_safe=False,
)
