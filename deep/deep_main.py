import time
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

from torchtext import data

import util as util
from models import *


SEED = 1234
torch.manual_seed(SEED)

# CONFIGURE THESE PARAMETERS
DEVELOPING = True
# DEVELOPING = False
WHICH_TASK = "response" # also can be "response"

if DEVELOPING:
    # in order: train, validation, test
    files = ["../data/complaints_3k.csv", \
                "../data/complaints_500.csv", \
                "../data/complaints_1k.csv"]
    BATCH_SIZE = 7
    MAX_VOCAB_SIZE = 5000
else:
    # in order: train, validation, test
    files = ["../data/full_training_set.csv", \
                "../data/full_validation_set.csv", \
                "../data/full_testing_set.csv"]    
    BATCH_SIZE = 64
    MAX_VOCAB_SIZE = 25000
    # TO DO: make full files
    pass

USE_CUDA = False
INPUT_DIM = MAX_VOCAB_SIZE + 2 # this is janky
NUM_EPOCHS = 1
GRAD_CLIP = 1



def load_and_tokenize_data(path, TEXT, LABEL, which_task):
    '''
    turn csv of complaints -> pytorch data object

    Takes: 
    - file path to data csv
    - TEXT field defini sction object
    - LABEL field definition object
    - string denoting which field is label ("response" or "product")
    Returns: 
    - Tabular dataset type
    '''

    # define fields
    if which_task == "response":
        data_fields = [('date_received', None),
                    ('product', None),
                    ('sub-product', None),
                    ('issue', None),
                    ('sub-issue', None),
                    ('narrative', TEXT), # note this is the field name, not colname in csv
                    ('company_public_response', None),
                    ('company', None),
                    ('state', None),
                    ('zip_code', None),
                    ('tags', None),
                    ('consumer_consent_provided', None),
                    ('submitted_via', None),
                    ('date_sent_to_company', None),
                    ('label', LABEL), # ditto here
                    ('timely_response', None),
                    ('consumer_disputed', None),
                    ('complaint_id', None)]
    else:
        data_fields = [('date_received', None),
                    ('label', LABEL), # ditto here
                    ('sub-product', None),
                    ('issue', None),
                    ('sub-issue', None),
                    ('narrative', TEXT), # note this is the field name, not colname in csv
                    ('company_public_response', None),
                    ('company', None),
                    ('state', None),
                    ('zip_code', None),
                    ('tags', None),
                    ('consumer_consent_provided', None),
                    ('submitted_via', None),
                    ('date_sent_to_company', None),
                    ('company_response_to_consumer', None), 
                    ('timely_response', None),
                    ('consumer_disputed', None),
                    ('complaint_id', None)]

    return data.TabularDataset(path=path,
                               format='csv',
                               skip_header=True,
                               fields=data_fields)


def preprocess(which_task, train_file, val_file, test_file, max_vocab_size=MAX_VOCAB_SIZE):
    '''
    Load data and preprocess:
    - apply tokenization
    - one hot encode labels
    - build embeddings

    Takes:
    - string denoting which field is label ("response" or "product")
    - filename of training data csv
    - filename of validation csv
    - filename of testing csv
    - max vocab size
    Returns:
    - train data, validation data, test data object
    '''

    if which_task not in ["response", "product"]:
        print("preprocessing error: which field is the label?")
        raise ValueError

    # define text field objects with tokenization
    TEXT = data.Field(sequential=True, tokenize=util.tokenize, lower=True)

    # define label field with one hot encoded labels
    if which_task == "response":
        OneHotEncoder = data.Pipeline(convert_token=util.one_hot_encode_response)
        LABEL = data.LabelField(sequential=False, use_vocab=False, preprocessing=OneHotEncoder)
    else:
        OneHotEncoder = data.Pipeline(convert_token=util.one_hot_encode_product)
        LABEL = data.LabelField(sequential=False, use_vocab=False, preprocessing=OneHotEncoder)


    train_data = load_and_tokenize_data(train_file, TEXT, LABEL, which_task)
    valid_data = load_and_tokenize_data(val_file, TEXT, LABEL, which_task)
    test_data = load_and_tokenize_data(test_file, TEXT, LABEL, which_task)

    # create embeddings from training data
    TEXT.build_vocab(train_data, max_size=max_vocab_size)
    LABEL.build_vocab(train_data)
    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

    return train_data, valid_data, test_data


def optimize_params(parameters, train_iter, val_iter):
    '''
    Find model parameters with the lowest validation error

    - Extract narrative word list and label from the batch
    - Pass the hidden state vector from output of previous batch as the initial hidden vector for
        the current batch, detaching
    - Zero out the model gradients to reset backpropagation for current batch
    - Call forward propagation to get output and final hidden state vector.
    - Compute loss
    - Run back propagation to set the gradients for each model parameter.
    - Clip the gradients that may have exploded. 
    - Evaluate model on the validation set periodically

    Takes: 
    - parameter dict
    Returns: 
    - best model
    - average loss on validation set
    '''

    print("Training model with parameters:")
    print(parameters)

    model = RNNModel(*list(parameters.values()))
    if USE_CUDA:
        model = model.cuda()

    # TO DO: maybe don't want to set these here
    learning_rate = 0.01 # 0.001 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    last_batch_size = BATCH_SIZE

    val_losses = []
    best_model = None
    start_time = time.time()

    print("begin training at", start_time)

    for epoch in range(NUM_EPOCHS):
        model.train()
        it = iter(train_iter)
        hidden = model.initHidden()
        for i, batch in enumerate(it):

            # extract narrative and label for batch
            batch_text = batch.narrative
            target = batch.label

            # drop last batch if it is too short
            # happens when number of narratives not divisible by batch size
            if i > 0 and batch_text.shape[1] != last_batch_size:
                break

            # if using a CUDA, put text on CUDA
            if USE_CUDA:
                batch_text = batch_text.cuda()
            
            # zero out gradients for current batch + call forward
            model.zero_grad()
            decoded, hiddenn = model(batch_text, hidden)

            # detach  hidden layers
            if model.rnn_type == "LSTM":
                hidden = hiddenn[0].detach(), hiddenn[1].detach()
            else:
                hidden = hiddenn.detach()

            # keep track of batch size
            last_batch_size = batch.batch_size

            # compute cross entropy loss 
            loss = model.loss_fn(decoded, target)

            # print batch loss every 500 iterations
            if i % 500 == 0:
                print(F"\t \t loss at {i}th", loss)

            # backpropagation + clip gradients + one step
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = GRAD_CLIP) 
            optimizer.step()
        

            # evaluate model every 1000 iterations
            if i % 10000 == 0:
                
                # compute loss, see if this is best model, append loss to loss list
                current_loss = model.evaluate(val_iter, BATCH_SIZE)
                
                if len(val_losses) == 0 or current_loss < min(val_losses):
                    best_model = model
                
                val_losses.append(current_loss)
            
    print("Training complete.")
    print("Training time: ", time.time() - start_time)
    
    return best_model, time.time() - start_time


def run_model(which_task, model_params, train_iter, valid_iter, test_iter, save=False, model_file=''):
    '''
    Trains single model w/ given parameters and evaluates on test

    Takes:
    - string denoting which field is label ("response" or "product")
    - dict of model parameters
    - train iterable of batches
    - validation iterable of batches
    - test iterable of batches
    - filename for model state dict
    - boolean to turn off saving model state dict
    '''

    best_model, train_time = optimize_params(model_params, train_iter, valid_iter)
    
    # compute loss on test set
    test_loss = best_model.evaluate(test_iter, BATCH_SIZE)
    print("Loss of best model on testing set:", test_loss)

    # save state
    if save:
        optimized_dict = best_model.state_dict()
        util.save_model(optimized_dict, model_file)

    return best_model, train_time, test_loss


if __name__ == "__main__":
    

    '''
    GET DATA PREPARED
    (don't want to run this more than 1x even with multiple models)
    '''
    train_data, valid_data, test_data = preprocess(WHICH_TASK, *files)

    train_iter, valid_iter, test_iter = data.BucketIterator.splits( \
    (train_data, valid_data, test_data), \
    sort_key = lambda x: len(x.narrative), \
    sort_within_batch=False, \
    batch_size = BATCH_SIZE) 

    iters = (train_iter, valid_iter, test_iter)

    '''
    DO MODEL RUNS
    '''


    if WHICH_TASK == "response":
        company_response_parameters = {
            "model_type": "LSTM", \
            "vocab_size": INPUT_DIM, \
            "embedding_size": 40, \
            "hidden_size": 50, \
            "num_layers": 2, \
            "n_categories": 5, \
            "dropout": 0.5
        }
    elif WHICH_TASK == "product":
        parameters = {
            "model_type": "LSTM", \
            "vocab_size": INPUT_DIM, \
            "embedding_size": 40, \
            "hidden_size": 50, \
            "num_layers": 2, \
            "n_categories": 18, \
            "dropout": 0.5
        }

    best_model, train_time, test_loss = run_model(WHICH_TASK, company_response_parameters, *iters, save=True, model_file='trained_model.pt')
