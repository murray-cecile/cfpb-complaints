'''
DEFINE NEURAL NETWORK MODEL CLASSES
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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

        # take mean of words in each narrative?
        narrative = torch.mean(embedded, dim=0)

        # pass through linear + ReLU layers
        z1 = self.linear1(narrative)
        h1 = self.relu(z1)
        z2 = self.linear2(z1)
        y_tilde = self.relu(z2)

        return y_tilde


class RNNModel(nn.Module):
    '''
    Define RNN model class
    Source: Homework 3
    
    Parameters needed to initialize a new instance:
    - type of RNN to train: text string, either LSTM or GRU
    - number of tokens
    - number of input dimensions
    - hidden dimension
    - number of layers desired
    - dropout

    Evaluation
    '''
    
    def __init__(self, rnn_type, vocab_size, embed_size, hidden_size, n_layers, n_tag, dropout=0.5):
        ''' Initialize the following layers:
            - Embedding layer/encoder
            - Recurrent neural network layer (LSTM, GRU)
            - Linear decoding layer to map from hidden vector to the vocabulary
            - Optionally, dropout layers.  Dropout layers can be placed after 
              the embedding layer or/and after the RNN layer. 
            - Optionally, initialize the model parameters. 
            
            Initialize a loss function
            
            Create attributes where model will store training time, loss info
            
        '''
        super(RNNModel, self).__init__()
        
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embed_size)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size, n_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(embed_size, hidden_size, n_layers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, n_tag)

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_tag = n_tag
        
        # loss function
        self.loss_fn = nn.MultiLabelSoftMarginLoss()

    def initHidden(self):
        '''
        Initialize a hidden vector (None/zeroes) 
        '''
        return None
        
    def forward(self, input, hidden0):
        ''' 
        Run forward propagation for a given minibatch of inputs:
        process through the embedding, RNN, and the decoding layer.
        
        Takes: input text, hidden vector (tuple)
        Returns: decoded probability scores, hidden vector (tuple)

        '''
        
        embeds = self.encoder(input)
        output, hiddenn = self.rnn(embeds, hidden0) 
        decoded = self.decoder(output)

        return decoded, hiddenn
    
    def evaluate(self, data, global_batch_size):
        '''
        Evaluate model on data.
        
        Takes: data iterator object
        Returns: average cross entropy loss across all batches in data
        '''

        self.eval()
        it = iter(data)
        total_count = 0. 
        total_loss = 0. 
        last_batch_size = global_batch_size
        with torch.no_grad():

            # Initialize hidden vector
            hidden = self.initHidden() 
            
            for i, batch in enumerate(it):
                                        
                # extract narrative and label for batch
                batch_text = batch.narrative
                target = batch.label

                # drop last batch if it is too short
                # happens when number of narratives not divisible by batch size
                if i > 0 and batch_text.shape[1] != last_batch_size:
                    break
            
                # zero out gradients for current batch and call forward propagation
                self.zero_grad()
                decoded, hiddenn = self(batch_text, hidden)

                if self.rnn_type == "LSTM":
                    hidden = hiddenn[0], hiddenn[1]
                else:
                    hidden = hiddenn

                # keep track of batch size
                last_batch_size = batch.batch_size

                # get loss for batch 
                loss = self.loss_fn(decoded, target)
                total_loss += loss * decoded.shape[0]
                
                # count number of narratives in batch
                total_count += decoded.shape[0]

        loss = total_loss / total_count
        self.train()
        return loss
