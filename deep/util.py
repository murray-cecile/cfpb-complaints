import random
import copy
import torch
import sklearn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

from torchtext import data

from models import RNNModel

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


def one_hot_encode_response(x):
    '''
    Converts string company response into one hot encoded label

    Takes: label string
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


def one_hot_encode_product(x):
    '''
    Converts string product category into one hot encoded label

    Takes: label string
    Returns: list with 1 in position corresponding to label 
    '''

    # list of product types
    product_types = ['Debt collection', \
                    'Credit reporting, credit repair services, or other personal consumer reports', \
                    'Vehicle loan or lease', \
                    'Mortgage', \
                    'Credit card or prepaid card', \
                    'Money transfer, virtual currency, or money service', \
                    'Student loan', \
                    'Checking or savings account', \
                    'Payday loan, title loan, or personal loan', \
                    'Credit card', \
                    'Consumer Loan', \
                    'Payday loan', \
                    'Bank account or service', \
                    'Credit reporting', \
                    'Prepaid card', \
                    'Other financial service', \
                    'Money transfers', \
                    'Virtual currency']

    encoded = [1 if x == i  else 0 for i in product_types]
    
    if sum(encoded) == 0:
        print("Unknown product type:")
        print(x)
        raise ValueError

    return encoded


'''
MODEL EVALUATION
'''

def multiclass_accuracy(preds, y):
    '''
    Compute accuracy for multiclass classification

    Takes: 
    - predictions matrix: the output of decoded 
    - true label matrix: each row is a one hot encoded vector corresponding
                         to the true class
    Returns: accuracy score
    '''

    # assign label based on max predicted value, compare index
    predicted = torch.softmax(preds, dim=2).argmax(dim=2, keepdim=True).squeeze()
    true = y.max(dim=1, keepdim=True).indices.squeeze()
    true = true.repeat(predicted.shape[0], 1, 1).view(predicted.shape).squeeze()
    
    correct = (predicted == true).float()  
    acc = correct.sum(dim=1) / correct.shape[1]

    return acc.mean()

def compute_accuracy(model, data):
    '''
    Compute multiclass accuracy for the model given a data iterable

    Takes: data iterable object
    Returns: multiclass accuracy score
    '''

    model.eval()
    it = iter(data)
    total_count = 0. 
    total_acc = 0. 
    last_batch_size = BATCH_SIZE
    with torch.no_grad():

        # Initialize hidden vector
        hidden = model.initHidden() 
        
        for i, batch in enumerate(it):
                                    
            # extract narrative and label for batch
            batch_text = batch.narrative
            target = batch.label

            # drop last batch if it is too short
            # happens when number of narratives not divisible by batch size
            if i > 0 and batch_text.shape[1] != last_batch_size:
                break

            # zero out gradients for current batch and call forward propagation
            model.zero_grad()
            decoded, hiddenn = model(batch_text, hidden)

            if model.rnn_type == "LSTM":
                hidden = hiddenn[0], hiddenn[1]
            else:
                hidden = hiddenn

            # keep track of batch size
            last_batch_size = batch.batch_size
            
            # get average loss for batch 
            acc = multiclass_accuracy(decoded, target)
            print("acc is", acc)

            total_acc += acc 
            total_count += 1
        
    final_acc = total_acc / total_count
    model.train()
    return final_acc
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
    trained_model = RNNModel(*list(params.values()))
        
    path = F"./models/{filename}" 
    trained_model.load_state_dict(torch.load(path), strict=False)
    return trained_model

