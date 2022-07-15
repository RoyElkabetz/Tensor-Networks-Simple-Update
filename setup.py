from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='tensor_networks_simple_update',
    version='0.0.1',
    description='An independent package for Tensor-Networks Simple-Update simulations.',
    url="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update",
    author="Roy Elkabetz",
    author_email="elkabetzroy@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # py_modules=['tensor_network', 'simple_update', 'structure_matrix_constructor', 'utils', 'ncon', 'examples'],
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        "numpy>=1.19.5",
        "matplotlib>=3.2.0",
        "scipy>=1.6.2",
    ],
    extras_require={
        "dev": [
            "pytest>=3.7",
        ],
    },

)
