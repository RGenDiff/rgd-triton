from setuptools import setup, find_packages

setup(
    name='RGD-Triton',
    version='0.1.0',
    description='Collection of Triton operators for transformer models.',
    packages=find_packages(),
    install_requires=[
        'torch>=2.5.1',
        'triton>=3.1.0',
    ],
    python_requires='>=3.9',
)
