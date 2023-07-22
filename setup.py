#!/usr/bin/env python3
from setuptools import setup, find_packages
from glob import glob


setup(
    name='mmapped_df',
    version='0.0.1',
    url='https://github.com/michalsta/mmapped_df',
    author='Michał Startek',
    author_email='author@gmail.com',
    description='Memory-mapped, on-disk pandas df',
    packages=find_packages(),    
    install_requires=['numpy', 'pandas', 'pyarrow'],
    scripts=glob("tools/*.py")
)
