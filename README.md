# Image-Classifier-Application
Transfer learning project [AI Programming with Python Nanodegree]

## Projcet Overview
### Project Description
- Train an image classifier to recognize different species of flowers.
- Build a command line application of the image classifier.

### Project Procedure
- Load and preprocess the data
- Build network
  - Load the pretrained model: vgg16
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

### Train a new network with train3.py
### Predict flower name with predict3.py
In a terminal or command window, run one of the following commands:

```bash
ipython notebook dlnd_tv_script_generation.ipynb
```  
or
```bash
jupyter notebook dlnd_tv_script_generation.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data
[Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
```bash
flower/
```
A 102 category dataset, consisting of 102 flower categories.
