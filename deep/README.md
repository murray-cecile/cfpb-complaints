# Deep learner classifiers

This subdirectory contains the code to create and run RNN models for the two prediction tasks.

- `RUN.ipynb` is a notebook that essentially implements `deep_main.py` for Amazon Sagemaker

- `deep_main.py` does the following: 
    - loads the data, tokenizes narratives, one-hot-encodes labels
    - creates batches for training, validation, and test
    - defines a model object instance with specific hyperparameters
    - optimizes the hyperparameters
    - evaluates model performance on the test set


- `models.py` contains the model classes
    - WordEmbedAvgLinear is a simple linear model with a ReLU that averages word-level embeddings for each narrative.
    - RNNModel implements LSTM or GRU with word-level embeddings, a linear decoder, and optional dropout

- `util.py` contains a suite of helper functions:
    - tokenization 
    - one-hot encoding labels for each prediction task
    - additional evaluation functions
    - utilities for saving and loading model `state_dict()`

### How to run models

Users should only need to modify `deep_main.py` in order to run models. The module can be configured with the following options:

1. Set `DEVELOPING=True` to run on smaller datasets.

2. Set `WHICH_TASK="response"` to predict company response and `WHICH_TASK="product"` to predict product type.

3. Save the optimized state dictionary after training is complete by specifying `save=True` and a model filename in the call to `run_model` at the end of the file, like so:

```
best_model, train_time, test_loss = run_model(WHICH_TASK, parameters, *iters, save=True)
```

