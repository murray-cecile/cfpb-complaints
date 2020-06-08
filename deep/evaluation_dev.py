    
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import util 
from models import *

BATCH_SIZE = 64

# TEXT = data.Field(sequential=True, tokenize=util.tokenize, lower=True)
# TEXT.build_vocab(train_data, max_size=25000)


def compute_accuracy(model, data):
    '''
    IN DEVELOPMENT
    Compute multiclass accuracy of the model

    Takes: data iterator object
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
        
            # print("batch_text shape is", batch_text.shape)
            # print(batch_text)
            # print("target shape is", target.shape)
            # print(target)

            # zero out gradients for current batch and call forward propagation
            model.zero_grad()
            decoded, hiddenn = model(batch_text, hidden)

            if model.rnn_type == "LSTM":
                hidden = hiddenn[0], hiddenn[1]
            else:
                hidden = hiddenn

            # keep track of batch size
            last_batch_size = batch.batch_size

            # reweight loss - IS THIS RIGHT?
            # print("decoded shape", decoded.shape)
            # chunks_in_batch, batch_size, C = decoded.shape
            # N = words_in_batch * batch_size
            
            # get average loss for batch 
            acc = util.multiclass_accuracy(decoded, target)
            print("loss is", acc)

            total_acc += acc 
            total_count += 1
        
    final_acc = total_acc / total_count
    model.train()
    return final_acc

