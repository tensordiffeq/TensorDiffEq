import setuptools
import os
import sys

if sys.platform == 'darwin':
    os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    install_requires = [x.strip() for x in f.readlines()]

setuptools.setup(
    name="tensordiffeq",
    version="0.1.6.3",
    author="Levi McClenny",
    author_email="levimcclenny@tamu.edu",
    description="Distributed PDE Solver in Tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tensordiffeq/tensordiffeq",
    download_url="https://github.com/tensordiffeq/tensordiffeq/tarball/v0.1.6.3",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires='>=3.6',
)
