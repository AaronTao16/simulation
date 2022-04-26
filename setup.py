from distutils.core import setup
from setuptools import find_packages

install_requires = [
    'bokeh>=0.13',
    'expiringdict>=1.1.4',
    'injector>=0.16.2',
    'joblib>=0.13.2',
    'keras>=2.3',
    'mmh3~=3.0.0',
    'numpy',
    'selenium>=3.141.0',
    'scikit-multiflow>=0.3.0',
    'spacy>=2.2',
    'tqdm>=4.19',
]

test_deps = [
    'pytest',
]

setup(
    name='simu',
    packages=find_packages(),
    license='MIT',
    install_requires=install_requires,
    tests_require=test_deps,
    extras_require=dict(
        test=test_deps,
    ),
)
