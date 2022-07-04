from setuptools import setup, find_packages
setup(
    name="singular_library",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "scipy",
        "numpy"
    ],
)
