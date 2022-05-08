# Neuroevolution - Evolution of convolutional neural networks

#### Author: Alexander Polok ([xpolok03@fit.vutbr.cz](mailto:xpolok03@fit.vutbr.cz))

#### Date: 8.5.2022

Implementation of Hierarchical Representations for Efficient Architecture Search (Liu et
al., [2017](https://doi.org/10.48550/arXiv.1711.00436)) and
Evolving Deep Convolutional Neural Networks for Image Classification (Sun et
al., [2020](https://doi.org/10.1109/TEVC.2019.2916183)).

Project focuses on analysis of several ways to encode Neural network architecture to the genotype. Random,
evolutionary and genetic search algorithms are compared with Flat, Hierarchical and Variable length
representations. 

## Installation

Change the current working directory to the root of the project.

```bash
cd __PROJECT_ROOT__
```

### Configure anaconda environment

Create new environment with [anaconda distribution](https://www.anaconda.com/) and activate it.

```bash
conda create -n Neuroevolution python=3.9 --yes
conda activate Neuroevolution
```

### Install required packages

```bash
pip install -r requirements.txt
```

## How to run

Program could be run with default arguments simply as:

```bash
python main.py
```

Several arguments are provided, to list all with their descriptions run:

```bash
python main.py -h 
```

To automatically generate experiment configurations run one of files in configs folder:

```bash
python configs/generate_configs_flat.py
```

> **_NOTE:_** There is also available PBS script to run experiments on gpu cluster. Script create group of jobs with
> differenet configurations provided in `configs` directory. Execute with `qsub run_job.sh`.

## Cleanup

```bash
conda deactivate
conda remove --name Neuroevolution --all --yes
```
