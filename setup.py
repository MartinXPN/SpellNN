from setuptools import find_packages, setup

setup(
    name='SpellNN',
    version='0.0.1',
    description='Python package for neural spell checking',
    author='Martin Mirakyan',
    author_email='mirakyanmartin@gmail.com',
    python_requires='>=3.6.0',
    url='https://github.com/MartinXPN/SpellNN',
    packages=find_packages(exclude=('tests',)),
    scripts=[
        'spellnn/data/wiki2text.py',
    ],
    install_requires=[
        'GitPython>=2.1.11',
        'fire>=0.2.1',
        'joblib>=0.13.2',
        'numpy>=1.13.3',
        'scikit-learn>=0.20.2',
        'tqdm>=4.31.1',
        'keras-contrib @ git+https://www.github.com/keras-team/keras-contrib@master',
        'wikiextractor @ git+https://github.com/attardi/wikiextractor.git@refs/pull/180/merge',
    ],
    extras_require={
        'tf': ['tensorflow==2.0.0rc1'],
        'tf_gpu': ['tensorflow-gpu==2.0.0rc1'],
        'analysis': ['matplotlib>=3.0.3', 'notebook>=6.0.1'],
        'test': ['coverage', 'pytest', 'pytest-cov'],
    },
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Full list of Trove classifiers: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
