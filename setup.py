import pathlib
from setuptools import find_packages, setup
import os

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

def read_requirements():
    with open('requirements.txt', 'r') as file:
        return [line.strip() for line in file if not line.startswith('#')]

dev_requires = [
    "black>=22.1.0",
    "importlib-resources>=5.4.0",
    "ipython>=7.31.0",
    "notebook>=7.0.4",
    "pre-commit>=2.17.0",
    "seaborn>=0.12.2",
]
test_requires = []
extras_require = {  # Optional
    "dev": dev_requires,
    "test": test_requires,
}

setup_kwargs = {
    "name": "sharcs",
    "version": "0.1.0",
    "description": "SHARCS: Routing to sub-networks with varying widths for sample adaptive inference efficiency.",
    "long_description": long_description,
    "author": "SHARCS team",
    "author_email": None,
    "maintainer": None,
    "maintainer_email": None,
    "url": None,
    # "package_data": package_data,
    "package_dir": {"": "src"},
    "packages": find_packages("src"),
    "install_requires": read_requirements(),
    "extras_require": extras_require,
    "python_requires": ">=3.8,<4.0",
}

setup(**setup_kwargs)