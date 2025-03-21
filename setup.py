from setuptools import setup, find_packages

setup(
    name='core',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        "torch",
        "torcheeg",
        "braindecode",
        "mne",
        "mne-bids",
        "numpy",
        "pandas",
        "tqdm",
        "scikit-learn",
    ]
)