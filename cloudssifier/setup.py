from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['absl-py', 'tensorflow>=1.11'] # Uncomment this for cpu runs and comment out for gpu.
# REQUIRED_PACKAGES = ['absl-py'] # Uncomment this for gpu runs and comment out for cpu.

setup(
  name='cloudssifier',
  version='0.1',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  description='Simple classifier trained on the cloud!',
)
