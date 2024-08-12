# AutoCellLabeler

This package contains code for producing training data for the AutoCellLabeler network, and post-processing code for interpreting its outputs. The actual AutoCellLabeler network architecture is implemented in [the `pytorch-3dunet` package](https://github.com/flavell-lab/pytorch-3dunet).

## Setup

To install, download the source code and install with `pip`:

```
git clone git@github.com:flavell-lab/AutoCellLabeler
cd AutoCellLabeler
pip install .
```

The name of the installed package is `autolabel`. You will also want to install [the `pytorch-3dunet` package](https://github.com/flavell-lab/pytorch-3dunet) using its setup instructions.

## Create training data

To create training, validation, and testing data for AutoCellLabeler from a set of multi-spectral images and human labels in CSV format, see the `make_AutoCellLabeler_input.ipynb` notebook in this package.

To use the training data in [our preprint](https://doi.org/10.1101/2024.07.18.601886), please refer to [our Dropbox](https://www.dropbox.com/scl/fo/ealblchspq427pfmhtg7h/ALZ7AE5o3bT0VUQ8TTeR1As?rlkey=1e6tseyuwd04rbj7wmn2n6ij7&e=3&st=ybsvv0ry&dl=0), under the `AutoCellLabeler/train_val_test_data` directory. (You will have to run the `make_AutoCellLabeler_input.ipynb` notebook to format this data.)

## Train and run AutoCellLabeler

For training and running AutoCellLabeler, please refer to [the `pytorch-3dunet` package](https://github.com/flavell-lab/pytorch-3dunet). 

The model weights used in [our preprint](https://doi.org/10.1101/2024.07.18.601886) are available in [our Dropbox](https://www.dropbox.com/scl/fo/ealblchspq427pfmhtg7h/ALZ7AE5o3bT0VUQ8TTeR1As?rlkey=1e6tseyuwd04rbj7wmn2n6ij7&e=3&st=ybsvv0ry&dl=0), under the `AutoCellLabeler/model_weights` directory.

## Post-process AutoCellLabeler output and evaluate performance

To post-process AutoCellLabeler output, including exporting it to CSV files, and comparing it to human labels for accuracy evaluation, please see the `parse_AutoCellLabeler_output.ipynb` notebook in this package.

The additional testing datasets used in [our preprint](https://doi.org/10.1101/2024.07.18.601886) are available in [our Dropbox](https://www.dropbox.com/scl/fo/ealblchspq427pfmhtg7h/ALZ7AE5o3bT0VUQ8TTeR1As?rlkey=1e6tseyuwd04rbj7wmn2n6ij7&e=3&st=ybsvv0ry&dl=0), under the `AutoCellLabeler/noisy_immobilized_eval` directory for the low-SNR TagRFP-only immobilized images, and the `AutoCellLabeler/freely_moving_eval` for the low-SNR TagRFP-only freely-moving images. (You will have to run the `make_AutoCellLabeler_input.ipynb` notebook to format this data.)

## Citation

To cite this work, please refer to this article:

#### Deep Neural Networks to Register and Annotate the Cells of the *C. elegans* Nervous System
Adam A. Atanas, Alicia Kun-Yang Lu, Jungsoo Kim, Saba Baskoylu, Di Kang, Talya S. Kramer, Eric Bueno, Flossie K. Wan, Steven W. Flavell

bioRxiv 2024; doi: https://doi.org/10.1101/2024.07.18.601886

