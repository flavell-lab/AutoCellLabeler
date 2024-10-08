from setuptools import setup, find_packages

setup(
    name="autolabel",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["numpy", "pynrrd", "h5py", "tqdm", "openpyxl", "pandas", "matplotlib"],
    author="Adam Atanas",
    author_email="adamatanas@gmail.com",
    description="A Python package to automate NeuroPAL labeling via interfacing with a 3D-UNet.",
    license="MIT",
    url="http://github.com/flavell-lab/AutoCellLabeler"
)

