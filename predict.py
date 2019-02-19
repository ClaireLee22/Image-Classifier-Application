import train as t
# Import python modules
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
import os
import json
import argparse


def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using the aparse module. This function returns these argument as
    an ArgumentParser object.
    Parameters:
        None - use argparse module to create and store command line arguments object
    Returns:
        parse_args() - data structure that stores the ccommand line arguments object
    """
    predict_parser = argparse.ArgumentParser()
    # Non-optional argument
    predict_parser.add_argument('image_path', type=str, default="flowers/test/28/image_05214.jpg", help='image path')
    # Non-optional checkpoint 
    predict_parser.add_argument('checkpoint', type=str, default="checkpoint.pth", help='data_dir')
    # Choose top K 
    predict_parser.add_argument('--topk', type=int, default=5, help='display topK prediction')
    # Choose category list
    predict_parser.add_argument('--category_name', type=str, default= 'cat_to_name.json', help='category name')
    # Choose preprocessor
    predict_parser.add_argument('--device', type=str, default= 'cuda', help='choose CPU or GPU to train model')
    
    return predict_parser.parse_args()


def resize(img, newSize): 
    """
    resize the images where the shortest side is 256 pixels, keeping the aspect ratio.
    Parameters:
        img: images we will use as input for the model
        newSize: 256 pixels
    Returns:
        rimg: resized image
    """    
    if img.width > img.height:
        new_height = newSize
        new_width = int(new_height* img.width/img.height)
    else:
        new_width = newSize
        new_height = int(new_width* img.height/img.width)
    rimg = img.resize((new_width, new_height))
    return rimg


def crop(img, newSize):
    """
    crop out the center 224x224 portion of the image
    Parameters:
        img: resized images 
        newSize: 224 pixels 
    Returns:
        cimg: croped image
    """    
    left_margin = (img.width - newSize)/2
    bottom_margin = (img.height - newSize)/2
    right_margin = left_margin + newSize
    top_margin = bottom_margin + newSize
    cimg = img.crop((left_margin, bottom_margin, right_margin,top_margin))
    return cimg


def normalization(img):
    """
    normalize image to squish the values to be between -1 and 1.
    Parameters:
        img: croped images 
    Returns:
        nimg: croped image
    """    
    
    img = np.array(img)/255 # scale by 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    nimg = (img - mean)/std
    return nimg


def transpose(img):
    """
    PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. 
    Reorder dimensions using ndarray.transpose.
    Parameters:
        img: normalized images 
    Returns:
        timg: transposed image
    """    
    timg = img.transpose((2, 0, 1))
    return timg


def process_image(image_path):
    """ 
    Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    Parameters:
        image_path: directory for storing images
    Returns:
        timg: preprocessed image
    """
    # Process a PIL image for use in a PyTorch model
    from PIL import Image
    img = Image.open(image_path)
    rimg = resize(img, 256)
    cimg = crop(rimg, 224)
    nimg = normalization(cimg)
    timg = transpose(nimg)
    return timg

def label_mapping(category_name):
    """
    load in a mapping from category label to category name.
    Parameters: 
        category_name: a dictionary mapping the integer encoded categories to the actual names of the flowers
    Returns:
        flowers_to_name: flowers' names corresponding to their index
    """
    with open(category_name, 'r') as f:
        flower_to_name = json.load(f)
    return flower_to_name


def predict(image_path, model, topk, category_name):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    Parameters: 
        image_path: directory for storing images
        model: trained network
        topk: top k most probable classes
    Returns:
        topkprobs: topk probabilities, 
        topkclasses: topk classes
        top5knames: flowers' names corresponding to topk classes
    """
    # process image
    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
    
    with torch.no_grad():
        output = model.forward(img)
    
    # calculate the class probabilities(softmax) for img
    ps = torch.exp(output)
    prob, classes = ps.topk(topk)
    
    # flip model.class_to_idx
    idx_to_class = {value: key for key, value in model.class_to_idx.items()} 
    
    # convert tensor to list
    topkprobs = prob.numpy().tolist()[0] # become list (or it will be a list inside a list)
    topkclasses = classes.numpy().tolist()[0]  
    
    #label mapping
    flower_to_name = label_mapping(category_name)
    topknames = [flower_to_name[idx_to_class[num]] for num in topkclasses]
    
    return topkprobs, topkclasses, topknames


def print_result(image_path, model, topk, category_name):
    """
    print out prediction result, show top 5 classes and probabilities
    Parameters: 
        image_path: directory for storing images
        model: trained network
    Returns:
        None
    """
    topkprobs, topkclasses, topknames = predict(image_path, model, topk, category_name)
    print("The most likely image class and the probability is ({}, {:.3f})".format(topkclasses[0], topkprobs[0]))
    print("Below is the Top {} classes with probabilities:".format(topk))
    print("\trank\tClass\tClass Name\tProbability")
    print("\t==================================================") 
    for i in range(topk):
        print("\t{:4}\t{:5}\t{:10}\t{:.3f}".format(i+1, topkclasses[i], topknames[i], topkprobs[i]))

        
# Main program function defined below
def main():
    in_arg = get_input_args()
    model = t.load_checkpoint(in_arg.checkpoint)
    process_image(in_arg.image_path)
    predict(in_arg.image_path, model, in_arg.topk, in_arg.category_name)
    print_result(in_arg.image_path, model, in_arg.topk, in_arg.category_name)


# Call to main function to run the program
if __name__ == "__main__":
    main()
