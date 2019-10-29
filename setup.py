from setuptools import find_packages
from setuptools import setup

# Use this for the CPU example
# REQUIRED_PACKAGES = ['absl-py', 'tensorflow==2.0.0']

# Use this for the GPU example
REQUIRED_PACKAGES = ['absl-py']

setup(
    name='cloudssifier',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description='Simple classifier trained on the cloud!',
)
