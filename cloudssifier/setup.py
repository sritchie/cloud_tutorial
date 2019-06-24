from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['absl-py']

setup(
  name='cloudssifier',
  version='0.1',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  description='Simple classifier trained on the cloud!',
)
