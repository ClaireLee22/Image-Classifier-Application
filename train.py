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

# Define transforms for the training, validation, and testing sets
data_transforms = {
'train': transforms.Compose([transforms.RandomRotation(30),
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])]),
'valid': transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])]),
'test': transforms.Compose([transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])])}


# Store information of two pretrained_models
pretrained_models = {
    'alexnet' : {'model': models.alexnet(pretrained=True), 'input_size': 9216},
    'vgg16' : {'model': models.vgg16(pretrained=True), 'input_size': 25088}
}

  
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using the aparse module. This function returns these argument as
    an ArgumentParser object.
    
    Parameters:
        None - use argparse module to create and store command line arguments object
    
    Returns:
        parse_args() - data structure that stores the ccommand line arguments object
    """
    train_parser = argparse.ArgumentParser()
    # Non-optional argument
    train_parser.add_argument('data_dir', type=str, default="flowers/", help='data_dir')
    # Choose directory to save checkpoint
    train_parser.add_argument('--save_dir', type=str, default="checkpoint.pth", help='directory to save checkpoint')
    # Choose model architecture
    train_parser.add_argument('--arch', type=str, default= 'vgg16', help='choose pretrained model')
    # Choose learning rate
    train_parser.add_argument('--learning_rate', type=float, default= 0.001, help='learning rate for traing model')
    # Choose numbers of the hidden layers
    train_parser.add_argument('--hidden_layers', type=int, default= 4096, help='number of hidden layers')
    # Choose numbers of epochs
    train_parser.add_argument('--epochs', type=int, default= 10, help='epochs')
    # Choose preprocessor
    train_parser.add_argument('--device', type=str, default= 'cuda', help='choose CPU or GPU to train model')
    
    return train_parser.parse_args()


def load_image(data_dir):
    """ 
    Load the dataset and define the dataloaders
    Parameters:
        None 
    Returns:
        dataloader - data structure that stores image datasets for training, testing and validing
    """
    # Create a dictionary for files in directory
    image_dir = {
        'train' : data_dir + '/train',
        'valid' :  data_dir + '/valid',
        'test':  data_dir + '/test'}

    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform= data_transforms[x]) for x in ['train', 'test', 'valid']}
    # Using the image datasets and the trainforms, define the dataloaders
    dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 32, shuffle = True) for x in ['train', 'test', 'valid']}

    return dataloader, image_datasets


def build_Network(model, input_size , hidden_layers):
    """
    build a new network with pretrained model and new classifier
    Parameters:
        model: pretrained model from torchvision model
        input_size: input data size
        hidden_layers: numbers of hidden layer
    Returns:
        model - new model with new classifier
    """        
    # Freeze parameter so we don't backdrop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Network
    classifier = nn.Sequential(OrderedDict([
                      ('fcl', nn.Linear(input_size, hidden_layers)),
                      ('relu', nn.ReLU()),
                      ('dropout', nn.Dropout(p=0.5)),
                      ('fc2', nn.Linear(hidden_layers, 102)),
                      ('output', nn.LogSoftmax(dim=1))
                      ]))

    model.classifier = classifier
    return model
 
    
def validation(model, validloader, criterion):
    """
    measure the performance of the trained model using valid dataset in order to avoid overfiting
    Parameters:
        model: pretrained model with our new classifier
        validloader: take the valid dataset and return batches of images and corresponding labels
        criterion: loss function
    Returns:
        test loss - accumulate loss as testing valid dataset
        accuracy - calculate how many times get the prediction correct
    """        
    test_loss = 0
    accuracy = 0
    for images, labels in validloader:
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


def do_deep_learning(model, lr, trainloader, validloader, epochs, device):
    """
    train the model/classifer and provide option to use GPU to do the calculations
    Parameters:
        model: pretrained model with our new classifier
        lr: learning rate
        trainloader: take the train dataset and return batches of images and corresponding labels
        validloader: take the valid dataset and return batches of images and corresponding labels
        epochs: number of passes through the full training set
        device: cpu or cuda
    Returns:
        model - traind model and update weights and bias
    """        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
    print_every = 40
    steps = 0
    
    model.to(device.lower())

    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            # move input and label tensors to the GPU
            inputs, labels = inputs.to(device.lower()), labels.to(device.lower())
            # zero out all the gradient set on tensors
            optimizer.zero_grad()

            # forward and backward process
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                #turn off gradients for validation
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion)
                print ("Epoch: {}/{}...".format(e+1, epochs),
                       "Training Loss: {:.4f}..".format(running_loss/print_every),
                       "Validation Loss: {:.3f}..".format(test_loss/len(validloader)),
                       "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0
                model.train()
    return model


def check_accuracy_on_test(model, testloader):
    """
    do validation on the test set
    Parameters:
        testloader: take the test dataset and return batches of images and corresponding labels
    Returns:
        None
    """        
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuray of the network on the test images: %d %%' % (100 * correct / total))
    
    
def save_checkpoint(filepath, model, input_size, epochs, arch, hidden_layers, lr, image_datasets):
    """
    save trained network for later use or train more 
    Parameters:
        filepath: a directory to save checkpoint
        epochs: number of passes through the full training set.
        arch: pretrained model
        lr: learning rate
    Returns:
        None
    """
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'input_size': input_size,
                  'output_size': 102,
                  'hidden_layers': hidden_layers,
                  'learning_rate': lr,
                  'epochs': epochs,
                  'arch': arch,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx} 
    torch.save(checkpoint, filepath)
  

def load_checkpoint(filepath):
    """
    loads the checkpoint and rebuilds the model 
    Parameters:
        filepath: a directory to save checkpoint
    Returns:
        model: trained network we have saved in checkpoint
    """      
    checkpoint = torch.load(filepath)
    classifier = nn.Sequential(OrderedDict([
                      ('fcl', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'])),
                      ('relu', nn.ReLU()),
                      ('dropout', nn.Dropout(p=0.5)),
                      ('fc2', nn.Linear(checkpoint['hidden_layers'], checkpoint['output_size'])),
                      ('output', nn.LogSoftmax(dim=1))
                      ]))
    model= pretrained_models[checkpoint['arch'].lower()]['model']
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    print(model)
    return model
    
    
# Main program function defined below
def main():
    in_arg = get_input_args()
    dataloader, image_datasets = load_image(in_arg.data_dir)
    
    # load the pretrained model 
    pretrained_model = pretrained_models[in_arg.arch.lower()]['model']
    input_size = pretrained_models[in_arg.arch.lower()]['input_size']
    
    model = build_Network(pretrained_model, input_size , in_arg.hidden_layers)
    model = do_deep_learning(model, in_arg.learning_rate, dataloader['train'], dataloader['valid'], in_arg.epochs, in_arg.device)
    check_accuracy_on_test(model, dataloader['test'])
    save_checkpoint(in_arg.save_dir, model, input_size, in_arg.epochs, in_arg.arch, in_arg.hidden_layers, in_arg.learning_rate, image_datasets)
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
