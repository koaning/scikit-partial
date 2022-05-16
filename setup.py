from setuptools import setup, find_packages

base_packages = ["scikit-learn>=0.24.0"]

dev_packages = [
    "flake8>=3.6.0",
    "pytest>=4.0.2",
    "mktestdocs>=0.1.0",
    "pre-commit>=2.17.0",
    "interrogate>=1.5.0",
    "black>=22.1.0",
    "pandas>=1.0.0",
]


setup(
    name="skpartial",
    version="0.1.0",
    packages=find_packages(exclude=["notebooks"]),
    install_requires=base_packages,
    extras_require={"dev": dev_packages},
)
