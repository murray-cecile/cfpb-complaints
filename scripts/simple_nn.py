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

BATCH_SIZE = 64
MAX_VOCAB_SIZE = 25000


def load_and_split_data():
    '''
    DO NOT USE
    Manual approach to loading csv, split into train and validation
    '''

    df = pd.read_csv(TINY_TRAIN_FILE)

    df_cols = df[['Consumer complaint narrative',
                  'Company response to consumer']]. \
        rename(columns={'Consumer complaint narrative': 'text',
                        'Company response to consumer': 'label'})

    # tokenize into strings
    df_cols['text'] = df_cols['text'].str.split(" ")

    train_df, test_df = train_test_split(df_cols, random_state=SEED)
    train = [(n, r) for n, r in train_df.apply(lambda x: list(x), axis=1)]
    test = [(n, r) for n, r in test_df.apply(lambda x: list(x), axis=1)]

    return train, test


def tokenize(x): 
    '''
    Tokenizer for torchtext. Right now, just splits on spaces
    
    Takes: string
    Returns: list of strings
    '''
    
    return x.split(" ")

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
                   ('Consumer complaint narrative', TEXT),
                   ('Company public response', None),
                   ('Company', None),
                   ('State', None),
                   ('ZIP code', None),
                   ('Tags', None),
                   ('Consumer consent provided?', None),
                   ('Submitted via', None),
                   ('Date sent to company', None),
                   ('Company response to consumer', LABEL),
                   ('Timely response?', None),
                   ('Consumer disputed?', None),
                   ('Complaint ID', None)]

    return data.TabularDataset(path=path,
                                         format='csv',
                                         skip_header=True,
                                         fields=data_fields)


# from HW2 problem 3 notebook
def binary_accuracy(preds, y):
    """
    Return accuracy per batch
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


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
        self.loss_fn = nn.BCEWithLogitsLoss()  # TO DO: CHANGE THIS FOR MULTICLASS

        # Choose an optimizer
        self.optimizer = optim.Adam(self.model.parameters())

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
          # batch.text has the texts and batch.label has the labels.

            self.optimizer.zero_grad()

            predictions = self.model(batch.text).squeeze()
            loss = self.loss_fn(predictions, batch.label)
            accuracy = binary_accuracy(predictions, batch.label)

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

                predictions = self.model(batch.text).squeeze()
                loss = self.loss_fn(predictions, batch.label)
                acc = binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == "__main__":

        
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True)
    LABEL = data.Field(sequential=False, use_vocab=False)

    data_obj = load_and_tokenize_data()
    train_data, valid_data = data_obj.split(random_state=random.seed(SEED))

    # create embeddings
    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
    LABEL.build_vocab(train_data)

    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size = BATCH_SIZE)

    INPUT_DIM = len(TEXT.vocab)
    # #You can try many different embedding dimensions. Common values are 20, 32, 64, 100, 128, 512
    EMBEDDING_DIM = 20
    HIDDEN_DIM = 50
    OUTPUT_DIM = 6
    # #Get the index of the pad token using the stoi function
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = WordEmbAvg(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX)

    tm = Training_module(model)
    best_model = tm.train_model(train_iterator, valid_iterator)

    # tm.model = best_model
    # test_loss, test_acc = tm.evaluate(test_iterator)
    # print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

'''
RESOURCES:
https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/

'''