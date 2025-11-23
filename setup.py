"""
Setup script for Lung Nodule Detection Project
Installs the package and dependencies
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="lung-nodule-detection",
    version="1.0.0",
    description="Explainable 3D Deep Learning Framework for Early Lung Nodule Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mohammad Fawad Karimi",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "SimpleITK>=2.3.0",
        "nibabel>=5.1.0",
        "pydicom>=2.4.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "pandas>=2.0.0",
        "batchgenerators>=0.25",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.14.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
    ],
    entry_points={
        "console_scripts": [
            "lung-nodule-train=main:main",
        ],
    },
)

