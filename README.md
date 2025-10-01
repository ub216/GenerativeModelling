# Generative Modelling Toolkit

This repository provides a set of tools to quickly set up and experiment with generative modelling. It includes easy-to-use scripts and configuration files, enabling users to run experiments with various generative models with minimal setup.

## Features

- Modular codebase for generative models (Diffusion, VAE, etc.)
- Configurable experiments via YAML files in [`configs/`](configs/)
- Ready-to-use training and evaluation scripts
- Extensible support for datasets (see [`loaders/`](loaders/))
- Extensible loss and metric modules

## TODOs
- [x] Add support for diffusion network
- [x] Add support for flow matching
- [ ] Add support for GANs
- [ ] Add support for CFG
- [ ] Update ResNet based UNet to X-former

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/ub216/GenerativeModelling
cd GenerativeModelling
```

### 2. Set Up the Environment

#### Using Conda

```sh
conda create -n genmod python=3.10 -y
conda activate genmod
pip install -r requirements.txt
```

#### Using pip

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Your Experiment

Edit or create a config file in [`configs/`](configs/) (e.g., [`configs/diffusion.yaml`](configs/diffusion.yaml), [`configs/vae.yaml`](configs/vae.yaml)) to specify model, dataset, and training parameters.

### 4. Train a Model

Run the training script with your chosen config:

```sh
python train.py --config configs/diffusion.yaml
```

### 5. Evaluate a Model

After training, evaluate your model using:

```sh
python eval.py --config configs/diffusion.yaml
```

## Repository Structure

- [`train.py`](train.py): Script to train generative models.
- [`eval.py`](eval.py): Script to evaluate trained models.
- [`configs/`](configs/): YAML configuration files for experiments.
- [`models/`](models/): Model architectures.
- [`losses/`](losses/): Loss functions for training.
- [`metrics/`](metrics/): Evaluation metrics.
- [`loaders/`](loaders/): Dataset loaders.
- [`utils.py`](utils.py): Utility functions.

## Contributing

Feel free to open issues or submit pull requests to improve the toolkit!

## License

MIT License
