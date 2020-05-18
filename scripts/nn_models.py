'''
DEFINE NEURAL NETWORK MODEL CLASSES
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd


class WordEmbAvgLinear(nn.Module):
    '''
    Extremely simple model:
    - averages narrative embeddings
    - two layers consisting of a linear model + ReLU
    '''

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, pad_idx):

        super().__init__()

        # Define an embedding layer, a couple of linear layers, and
        # the ReLU non-linearity.
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, text):

        embedded = self.embedding(text)

        # take mean of words in each narrative: RECONSIDER?
        narrative = torch.mean(embedded, dim=0)

        # pass through linear + ReLU layers
        z1 = self.linear1(narrative)
        h1 = self.relu(z1)
        z2 = self.linear2(z1)
        y_tilde = self.relu(z2)

        return y_tilde


class WordEmbAvgRNN(nn.Module):
    '''
    RNN model:
    - averages narrative embeddings
    ???
    '''

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, pad_idx):

        super().__init__()

        # Define an embedding layer and RNN
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        # TO DO: DEFINE RNN

    def forward(self, text):

        embedded = self.embedding(text)

        # take mean of words in each narrative: RECONSIDER?
        narrative = torch.mean(embedded, dim=0)

        # TO DO: pass through RNN

        return y_tilde