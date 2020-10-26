import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    install_requires = [x.strip() for x in f.readlines()]

setuptools.setup(
    name="tensordiffeq", # Replace with your own username
    version="0.0.1",
    author="Levi McClenny",
    author_email="levimcclenny@tamu.edu",
    description="Distributed PDE Solver in Tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/levimcclenny/tensordiffeq",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
