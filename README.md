# Image-Classifier-Application
Transfer learning project [AI Programming with Python Nanodegree]

## Projcet Overview
### Project Description
- Train an image classifier to recognize different species of flowers.
- Build a command line application of the image classifier.

### Project Procedure
- Load and preprocess the data
- Build network
  - Load the pretrained model: vgg16/alexnet
  - Define a new classifer
- Train the classifer
- Test the network
- Make a prediction
- Save the checkpoint

### Project Results
- Train an classifier which can predict the class for an input flower image.
- Build the command line application
  - Allow users to choose different hyperparametrs to train their own classifers. 
  - Make predictions for input flower images using the classifer they have trained.

## Getting Started
### Prerequisites
- [PyTorch 4.0](https://pytorch.org/get-started/locally/)

### Train a new network with `train.py`

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu


### Predict flower name with `predict.py`

```bash
jupyter notebook dlnd_tv_script_generation.ipynb
```

### Data
Download [Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) 

A 102 category dataset, consisting of 102 flower categories.
