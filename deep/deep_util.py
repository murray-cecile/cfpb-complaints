import random
import copy
import torch
import sklearn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

from torchtext import data

'''
PREPROCESSING
'''

def tokenize(x): 
    '''
    Tokenizer for torchtext. Right now, just splits on spaces
    
    Takes: string
    Returns: list of strings
    '''
    
    return x.split(" ")


def one_hot_encode_label(x):
    '''
    Converts string label into one hot encoded label

    Takes: string
    Returns: list with 1 in position corresponding to label 
    '''

    if x == "Closed with explanation":
        return [1, 0, 0, 0, 0]
    elif x == "Closed with non-monetary relief":
        return [0, 1, 0, 0, 0]
    elif x == "Closed with monetary relief":
        return [0, 0, 1, 0, 0]
    elif x == "Untimely response":
        return [0, 0, 0, 1, 0]
    elif x == "Closed":
        return [0, 0, 0, 0, 1]
    else:
        print("Unexpected class label in one-hot encoding")
        print(x)
        raise ValueError

'''
MODEL TRAINING
'''

# should optimize_params() go here?

'''
MODEL EVALUATION
'''

def multiclass_accuracy(preds, y):
    '''
    Compute accuracy for multiclass classification

    Takes: 
    - predictions matrix: each row contains predicted score for each class
    - true label matrix: each row is a one hot encoded vector corresponding
                         to the true class
    Returns: accuracy score
    '''

    # assign label based on max predicted value, compare index
    predicted = preds.max(dim=1, keepdim=True).indices
    true = y.max(dim=1, keepdim=True).indices
    
    correct = (predicted == true).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


'''
LOADING/SAVING
'''

def save_model(model_dict, filename):
    '''
    Save best model state dictionary 

    Takes:
    - model state_dict() object
    - string filename
    Returns: None
    '''
    path = F"./models/{filename}" 
    torch.save(model_dict, path)


def load_model(params, filename):
    '''
    Load trained model from saved state dictionary 

    Takes:
    - dictionary of parameters
    - string filename 
    Returns:
    - model object
    '''
    best_model = RNNModel(*list(parameters.values()))
    
    if USE_CUDA:
        best_model = best_model.cuda()
    
    path = F"./models/{filename}" 
    best_model.load_state_dict(torch.load(path), strict=False)
    return best_model

