# dg-linear-time

This repository contains the code for the paper ["Between Linear and Sinusoidal: Rethinking the Time Encoder in Dynamic Graph Learning"](https://arxiv.org/abs/2504.08129). Much of the code is inherited and refactored from [DyGLib](https://github.com/yule-BUAA/DyGLib).

## Dataset Preparation

1. Create dataset directory

```
mkdir datasets
```

2. Create two subdirectories under `datasets`

```
cd datasets/
mkdir original
mkdir preprocessed
```

3. Download **UCI**, **Wikipedia**, **Enron**, **Reddit**, **LastFM**, and **USLegis** from this [link](https://zenodo.org/records/7213796#.Y1cO6y8r30o) to `datasets/original` and unzip them

## Build the Environment

We use Python 3.10, PyTorch 2.1.1, and CUDA 11.8

```
conda env create -f environment.yaml
```

## Run an Experiment

Run an experiment with pre-selected hyper-parameters:

```
python train.py -m experiment=uci_dygformer_linear seed=1,2,3,4,5 trainer.devices=[0]
```

Please find the config files corresponding to each experiment in `configs/experiments/`.
They are named in the format of `{dataset_name}_{model}_{time_encoder}`.
The yaml files for model selection under historical negative sampling have names end with `_historical_select`.

## Run a Hyper-Parameter Search

Redo a hyper-parameter search run:

```
python train.py -m experiment=uci_dygformer_linear hparams_search=dygformer seed=42 trainer.devices=[0] logger.wandb.group=uci-dygformer-linear-hparams
```

Simply add `hparams_search={model}` and `logger.wandb.group={dataset_name}-{model}-{time_encoder}-hparams`.
For **DyGFormer** and **DyGFormer-separate**, set `hparams_search=dygformer`.
Set `hparams_search=dygdecoder` for **DyGDecoder** and `hparams_search=tgat` for **TGAT**.
