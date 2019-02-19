# Image-Classifier-Application
Transfer learning project [AI Programming with Python Nanodegree]

## Projcet Overview
### Project Description
Use vgg16 pretrained model to train an image classifier to recognize different species of flowers. Then convert it into a command line application.

### Project Procedure
- Preprocess the data
  - Transform words to ids
  - Tokenize punctuation
- Create lookup table
- Build RNN
- Train RNN
- Generate TV script

### Project Results
- Generated TV script for a scene at Moe's Tavern successfully.
- Need to get more data to train to get a better results.


## Getting Started
### Prerequisites
- [PyTorch 4.0](https://pytorch.org/get-started/locally/)

### Run
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
```bash
data/simpsons/moes_tavern_lines.txt
```  
Part of the Simpsons dataset of scripts which consists of only the scenes in Moe's Tavern.
