from setuptools import find_packages
from setuptools import setup

setup(
    name='cloudssifier',
    version='0.1',
    install_requires=['absl-py'],
    extras_require={
        # Tensorflow lives in extras_require so that your local installation
        # doesn't clash with the install in the cloud.
        'local': ['tensorflow==1.14.*'],
        'tf2': ['tensorflow==2.0.*']
    },
    packages=find_packages(),
    description='Simple classifier trained on the cloud!',
)
