# AutoCellLabeler

This package contains code for producing training data for the AutoCellLabeler network, and post-processing code for interpreting its outputs. The actual AutoCellLabeler network architecture is implemented in [the `pytorch-3dunet` package](https://github.com/flavell-lab/pytorch-3dunet).

## Setup

To install, download the source code and install with `pip`:

```
git clone git@github.com:flavell-lab/AutoCellLabeler
cd AutoCellLabeler
pip install .
```

The name of the installed package is `autolabel`. You will also want to install the [the `pytorch-3dunet` package](https://github.com/flavell-lab/pytorch-3dunet) using its setup instructions.

## Create training data

To create training data from AutoCellLabeler from a set of multi-spectral images and human labels in CSV format, see the `make_AutoCellLabeler_input.ipynb` notebook in this package.

## Citation

To cite this work, please refer to this article:

#### Deep Neural Networks to Register and Annotate the Cells of the *C. elegans* Nervous System
Adam A. Atanas, Alicia Kun-Yang Lu, Jungsoo Kim, Saba Baskoylu, Di Kang, Talya S. Kramer, Eric Bueno, Flossie K. Wan, Steven W. Flavell

bioRxiv 2024; doi: https://doi.org/10.1101/2024.07.18.601886

