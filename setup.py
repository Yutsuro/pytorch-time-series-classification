from os import path
from codecs import open
from setuptools import setup, find_packages

package_name = "tisc"

root_dir = path.abspath(path.dirname(__file__))

def _requirements():
    return [name.rstrip() for name in open(path.join(root_dir, 'requirements.txt'), encoding="utf-8-sig").readlines()]

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=package_name,
    description='Simple model creation and training framework for time series classification in Pytorch',
    version='0.1.1',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Yutsuro/pytorch-time-series-classification',
    author='Yutsuro',
    author_email='Yuki@utsu.ro',
    license='Apache',
    keywords='Pytorch Time-Series-Classification LSTM BiLSTM Transformer Machine-Learning Deep-Learning AI Artificial-Intelligence',
    # packages=[package_name],
    packages=find_packages(),
    install_requires=_requirements(),
    classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)