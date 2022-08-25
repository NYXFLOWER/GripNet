from setuptools import setup

requirements = ["numpy", "pandas", "scikit-learn>=0.21.2", "scipy", "matplotlib"]

setup(
    name="gripnet",
    version="1.0",
    packages=["gripnet"],
    python_requires=">=3.7",
    install_requires=requirements,
    license="MIT License",
)
