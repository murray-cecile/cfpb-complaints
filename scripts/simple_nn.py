import random
import copy
import torch
import sklearn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

from torchtext import data
from sklearn.model_selection import train_test_split

SEED = 1234
torch.manual_seed(SEED)
TINY_TRAIN_FILE = "data/complaints_1k.csv"
TINY_TEST_FILE = "data/complaints_500.csv"

BATCH_SIZE = 64
MAX_VOCAB_SIZE = 25000


def tokenize(x): 
    '''
    Tokenizer for torchtext. Right now, just splits on spaces
    
    Takes: string
    Returns: list of strings
    '''
    
    return x.split(" ")


def encode_label(x):
    '''
    Converts string label into numeric label
    IS THIS BAD???

    Takes: string
    Returns: arbitrary numeric assignment
    '''

    if x == "Closed with explanation":
        # return [1, 0, 0, 0, 0]
        return 0
    elif x == "Closed with non-monetary relief":
        # return [0, 1, 0, 0, 0]
        return 1
    elif x == "Closed with monetary relief":
        # return [0, 0, 1, 0, 0]
        return 2
    elif x == "Untimely response":
        # return [0, 0, 0, 1, 0]
        return 3
    elif x == "Closed":
        # return [0, 0, 0, 0, 1]
        return 4
    elif x == "Closed with explanation":
        # return [0, 0, 0, 0, 0]
        return 5
    else:
        raise ValueError
        print("Unexpected class label in one-hot encoding")



def multiclass_accuracy(preds, y):
    """
    Return accuracy per batch
    """

    # assign label based on max predicted value, get index
    predicted = preds.max(dim=1).indices

    correct = (predicted == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def load_and_tokenize_data(path=TINY_TRAIN_FILE):
    '''
    turn csv of complaints -> pytorch data object

    Takes: file path to data csv
    Returns: 
    '''

    # define which fields we want
    data_fields = [('Date received', None),
                   ('Product', None),
                   ('Sub-product', None),
                   ('Issue', None),
                   ('Sub-issue', None),
                   ('narrative', TEXT), # note this is the field name, not colname in csv
                   ('Company public response', None),
                   ('Company', None),
                   ('State', None),
                   ('ZIP code', None),
                   ('Tags', None),
                   ('Consumer consent provided?', None),
                   ('Submitted via', None),
                   ('Date sent to company', None),
                   ('label', LABEL), # ditto here
                   ('Timely response?', None),
                   ('Consumer disputed?', None),
                   ('Complaint ID', None)]

    return data.TabularDataset(path=path,
                                         format='csv',
                                         skip_header=True,
                                         fields=data_fields)


# also from HW2 problem 3
class WordEmbAvg(nn.Module):
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

        # take mean of words in each review
        sentence = torch.mean(embedded, dim=0)

        # pass through linear + ReLU layers
        z1 = self.linear1(sentence)
        h1 = self.relu(z1)
        z2 = self.linear2(z1)
        y_tilde = self.relu(z2)

        return y_tilde


# also from HW2 problem 3
class Training_module():

    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()  # TO DO: RECONSIDER THIS

        # Choose an optimizer
        self.optimizer = optim.Adam(self.model.parameters()) # TO DO: RECONSIDER

    def train_epoch(self, iterator):
        '''
        Train the model for one epoch. For this repeat the following, 
        going through all training examples.
        1. Get the next batch of inputs from the iterator.
        2. Determine the predictions using a forward pass.
        3. Compute the loss.
        4. Compute gradients using a backward pass.
        5. Execute one step of the optimizer to update the model paramters.
        '''
        epoch_loss = 0
        epoch_acc = 0

        for batch in iterator:
          # batch.narrative has the texts and batch.label has the labels.

            self.optimizer.zero_grad()

            predictions = self.model(batch.narrative)
            loss = self.loss_fn(predictions, batch.label)
            accuracy = multiclass_accuracy(predictions, batch.label)

            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += accuracy.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def train_model(self, train_iterator, dev_iterator):
        """
        Train the model for multiple epochs, and after each evaluate on the
        development set.  Return the best performing model.
        """
        dev_accs = [0.]
        for epoch in range(9):
            self.train_epoch(train_iterator)
            dev_acc = self.evaluate(dev_iterator)
            print(
                f"Epoch {epoch}: Dev Accuracy: {dev_acc[1]} Dev Loss:{dev_acc[0]}")
            if dev_acc[1] > max(dev_accs):
                best_model = copy.deepcopy(self)
            dev_accs.append(dev_acc[1])
        return best_model.model

    def evaluate(self, iterator):
        '''
        Evaluate the performance of the model on the given examples.
        '''
        epoch_loss = 0
        epoch_acc = 0

        with torch.no_grad():

            for batch in iterator:

                predictions = self.model(batch.narrative)
                loss = self.loss_fn(predictions, batch.label)
                acc = multiclass_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == "__main__":

        
    # define preprocessing pipeline object
    NumericEncoder = data.Pipeline(convert_token=encode_label)

    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True)
    LABEL = data.LabelField(sequential=False, use_vocab=False, preprocessing=NumericEncoder)

    data_obj = load_and_tokenize_data(TINY_TRAIN_FILE)
    train_data, valid_data = data_obj.split(random_state=random.seed(SEED))
    test_data = load_and_tokenize_data(TINY_TEST_FILE)

    # create embeddings
    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
    LABEL.build_vocab(train_data)

    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort_key = lambda x: len(x.narrative),
        sort_within_batch=False,
        batch_size = BATCH_SIZE)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 20
    HIDDEN_DIM = 50
    OUTPUT_DIM = 6
    # #Get the index of the pad token using the stoi function
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = WordEmbAvg(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX)



    tm = Training_module(model)
    best_model = tm.train_model(train_iterator, valid_iterator)

    tm.model = best_model
    test_loss, test_acc = tm.evaluate(test_iterator)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

'''
RESOURCES:
https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/

'''