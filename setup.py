#!/usr/bin/env python3
from glob import glob

from setuptools import find_packages, setup


def get_polars():
    try:
        import cpuinfo
    except ModuleNotFoundError:
        return "polars"
    info = cpuinfo.get_cpu_info()
    if info["arch"] != "X86_64":
        return "polars"
    if "avx2" in info["flags"]:
        return "polars"
    else:
        return "polars-lts-cpu"


setup(
    name="mmapped_df",
    version="0.0.1",
    url="https://github.com/michalsta/mmapped_df",
    author="Michał Startek",
    author_email="author@gmail.com",
    description="Memory-mapped, on-disk pandas df",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "pyarrow", "numba", "h5py", "numba_progress"]
    + [get_polars()],
    scripts=glob("tools/*.py"),
)
