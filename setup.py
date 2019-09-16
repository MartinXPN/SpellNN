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
    install_requires=[
        'GitPython>=2.1.11',
        'fire>=0.1.3',
        'joblib>=0.13.2',
        'Keras>=2.2.4',
        'numpy>=1.16.1',
        'nltk>=3.4.5',
        'gensim>=3.8.0',
        'scikit-learn>=0.20.2',
        'tqdm>=4.31.1',
        'keras-contrib @ git+https://www.github.com/keras-team/keras-contrib@master',
    ],
    extras_require={
        'tf': ['tensorflow>=1.12.0'],
        'tf_gpu': ['tensorflow-gpu>=1.12.0'],
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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
