# From virtual Z gates to virtual Z pulses source code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17123422.svg)](https://doi.org/10.5281/zenodo.17123422)

This repository contains the source code required to reproduce and plot the data presented in:

C. K. Long and C. H. W. Barnes, From virtual z gates to virtual z pulses, 2025. arXiv: 2509.13453 \[quant-ph\]. https://arxiv.org/abs/2509.13453.

## Installation

The required Python packages can be installed by executing
```bash
pip install -r requirements.txt
pip install -e ./
```
in the root directory of this repository. To ensure reproducibility, all packages in ``requirements.txt`` are version pinned, and Python ``3.13.5`` should be used.

## Downloading the data

The data for the article can be found at:

Long, C. K., & Barnes, C. H. W. (2025). From virtual Z gates to virtual Z pulses data [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17113741

and can be downloaded to the correct directories for plotting by running
```bash
bash scripts/download_data_from_zenodo.sh
```

## Reproducing the data

Alternatively, all the data can be collected and plotted by executing
```bash
python scripts/collect_and_plot_all_data.py
```

To only collect the data you can execute
```bash
python scripts/data_collection/collect_all_data.py
```
in the root directory of this repository. Alternatively, if you only wish to collect the data for Figure ``X`` in our article, then you can execute
```bash
# Replace X with the figure number
python scripts/data_collection/figure_X.py 
```
in the root directory of this repository.

### Plotting the data

All the figures can be plotted from the collected data by executing
```bash
python scripts/plotting/plot_all_data.py
```
in the root directory of this repository. Alternatively, if you only wish to reproduce Figure ``X`` in our article then you can execute
```bash
# Replace X with the figure number
python scripts/plotting/figure_X.py 
```
in the root directory of this repository.

## Updates

This repository will only be updated to fix bugs that prevent reproducing the data presented in the article, to update citations, or if the article is updated. To ensure reproducibility of the article any bugs that are found that have impacted the data in the article will not be fixed unless our article is also updated.