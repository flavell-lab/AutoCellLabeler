from setuptools import setup, find_packages

setup(
    name="autolabel",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "pynrrd", "h5py", "tqdm", "openpyxl", "pandas"],
    author="Adam Atanas",
    author_email="adamatanas@gmail.com",
    description="A Python package to automate NeuroPAL labeling via interfacing with a 3D-UNet.",
    license="MIT",
    url="http://github.com/flavell-lab/autolabel"
)

