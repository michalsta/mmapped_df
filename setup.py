#!/usr/bin/env python3
from glob import glob

from setuptools import find_packages, setup


setup(
    name="mmapped_df",
    version="0.0.1",
    url="https://github.com/michalsta/mmapped_df",
    author="Michał Startek",
    author_email="author@gmail.com",
    description="Memory-mapped, on-disk pandas df",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "pyarrow", "numba", "h5py", "numba_progress"],
    extras_require={"polars": ["install-proper-polars"]},
    scripts=glob("tools/*.py"),
)
