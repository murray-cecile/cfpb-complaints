import time
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

from torchtext import data
from torch.nn.utils.rnn import pad_sequence

import deep_util as util
from deep_models import *


SEED = 1234
torch.manual_seed(SEED)

# TO DO: improve this structure
DEVELOPING = True
# DEVELOPING = False

if DEVELOPING:
    # in order: train, validation, test
    files = ["data/complaints_3k.csv", \
                "data/complaints_500.csv", \
                "data/complaints_1k.csv"]
    BATCH_SIZE = 8
    MAX_VOCAB_SIZE = 5000
else:
    BATCH_SIZE = 64
    MAX_VOCAB_SIZE = 25000
    # TO DO: make full files
    pass


def load_and_tokenize_data(path, TEXT, LABEL):
    '''
    turn csv of complaints -> pytorch data object

    Takes: 
    - file path to data csv
    - TEXT field definition object
    - LABEL field definition object
    Returns: 
    - Tabular dataset type
    '''

    # define which fields we want
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

    return data.TabularDataset(path=path,
                               format='csv',
                               skip_header=True,
                               fields=data_fields)


def preprocess(train_file, val_file, test_file, max_vocab_size=MAX_VOCAB_SIZE):
    '''
    Load data and preprocess:
    - apply tokenization
    - one hot encode labels
    - build embeddings

    Takes:
    - filename of training data csv
    - filename of validation csv
    - filename of testing csv
    - max vocab size
    Returns:
    - train data, validation data, test data object
    '''

    # define preprocessing pipeline object
    OneHotEncoder = data.Pipeline(convert_token=util.one_hot_encode_label)

    # define text and label field objects with preprocessing
    TEXT = data.Field(sequential=True, tokenize=util.tokenize, lower=True)
    LABEL = data.LabelField(sequential=False, use_vocab=False, preprocessing=OneHotEncoder)

    train_data = load_and_tokenize_data(train_file, TEXT, LABEL)
    valid_data = load_and_tokenize_data(val_file, TEXT, LABEL)
    test_data = load_and_tokenize_data(test_file, TEXT, LABEL)

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

            # debugging batch size issue
            # print("\t \t narrative shape", batch.narrative.shape)
            # print("\t \t hidden shape", hidden.shape)

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
            
#             print("\t \t decoded shape", decoded.shape)
#             print("\t \t target shape", target.shape)

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
            if i % 1000 == 0:
                
                # compute loss, see if this is best model, append loss to loss list
                current_loss = model.evaluate(val_iter)
                
                if len(val_losses) == 0 or current_loss < min(val_losses):
                    best_model = model
                
                val_losses.append(current_loss)
            
    print("Training complete.")
    print("Training time: ", time.time() - start_time)
    
    return best_model, time.time() - start_time


def run_model(model_params, train_iter, valid_iter, test_iter, save=False, model_file=''):
    '''
    Trains single model w/ given parameters and evaluates on test

    Takes:
    - dict of model parameters
    - train iterable of batches
    - validation iterable of batches
    - test iterable of batches
    - filename for model state dict
    - boolean to turn off saving model state dict
    '''

    best_model, train_time = optimize_params(model_params, train_iter, valid_iter)
    
    # compute perplexity
    perplexity = torch.exp(best_model.evaluate(test_iter))
    print("Perplexity of best model on testing set:", perplexity)

    # save state
    if save:
        optimized_dict = best_model.state_dict()
        util.save_model(optimized_dict, model_file)

    return best_model, train_time, perplexity

if __name__ == "__main__":
    

    '''
    GET DATA PREPARED
    (don't want to run this more than 1x even with multiple models)
    '''
    train_data, valid_data, test_data = preprocess(*files)

    train_iter, valid_iter, test_iter = data.BucketIterator.splits( \
    (train_data, valid_data, test_data), \
    sort_key = lambda x: len(x.narrative), \
    sort_within_batch=False, \
    batch_size = BATCH_SIZE) 

    iters = (train_iter, valid_iter, test_iter)

    '''
    DO MODEL RUNS
    '''

    # TO DO: do something better with these guys
    USE_CUDA = False
    INPUT_DIM = MAX_VOCAB_SIZE + 2 # this is janky
    NUM_EPOCHS = 1
    GRAD_CLIP = 1


    parameters = {
        "model_type": "LSTM", \
        "vocab_size": INPUT_DIM, \
        "embedding_size": 40, \
        "hidden_size": 50, \
        "num_layers": 2, \
        "n_categories": 5, \
        "dropout": 0.5
    #     "tie_weights": False # didn't implement this but we coulds
    }


    # this can get put into a loop, if it doesn't run insanely slowly
    best_model, train_time, perplexity = run_model(parameters, *iters, save=False)