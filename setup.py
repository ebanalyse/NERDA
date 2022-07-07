import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("src/NERDA/about.py") as f:
    v = f.read()
    for l in v.split("\n"):
        if l.startswith("__version__"):
            __version__ = l.split('"')[-2]

setuptools.setup(
    name="NERDA",
    version=__version__,
    author="Lars Kjeldgaard, Lukas Christian Nielsen",
    author_email="lars.kjeldgaard@eb.dk",
    description="A Framework for Finetuning Transformers for Named-Entity Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ebanalyse/NERDA",
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'transformers',
        'sklearn',
        'nltk',
        'pandas',
        'progressbar',
        'pyconll',
        'torchcrf'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest',
                   'pytest-cov'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True
    )
