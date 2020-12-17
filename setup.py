import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NERDA", 
    version="0.0.1",
    author="PIN",
    author_email="lars.kjeldgaard@eb.dk",
    description="Named Entity Recognition for DAnish based on Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ebanalyse/NERDA",
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'transformers==3.5.1',
        'tensorboard',
        'danlp',
        'sklearn',
        'pandas',
        'nltk'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True
    )