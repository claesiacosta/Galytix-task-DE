import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='Galytix - task',
    version='1.0',
    author='Claésia Costa', 
    packages=find_packages(),
    install_requires=required,
)