#==============================================================================#
# CFPB COMPLAINTS: CLASSIFY
#
# Nora Hajjar, Cecile Murray, Erika Tyagi
#==============================================================================#

'''
Runs the full classification loop. 

Required input: path to training and testing data
Output: The script will print a summary of all evaluated models as well as the
        highest AUC achieved.
'''


# import existing Python libraries
import argparse
# import sklearn
import datetime
import timeit
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestRegressor, AdaBoostClassifier,
                              BaggingClassifier, GradientBoostingClassifier)
from sklearn.utils.testing import ignore_warnings
from sklearn import metrics
from sklearn.metrics import (classification_report, precision_recall_curve,
                             roc_auc_score, plot_roc_curve)

# import project-specific Python libraries
import etl

@ignore_warnings(category=FutureWarning)


# TO DO: 
# REVAMP TRAINING INPUT FOR PYTORCH ARCHITECTURE 
# REPLACE CROSS VALIDATION WITH TRAIN/TUNE?

def find_best_model(models, parameters_grid, x_train, outcome_label):
    '''
    Cross-validation to find the best model, given parameters.
    Input:
        models (dict): dictionary initializing each scikit-learn model
        parameters_grid (dict): dictionary setting hyperparameter values
        x_train (df): processed training feature dataframe
        outcome_label (str): name of outcome label
    Output:
        best_model (obj): best model fit on full training data
    '''
    results_df =  pd.DataFrame(columns=('model_name',
                                        'parameters',
                                        'auc',
                                        'time_to_run'))
    max_auc = 0
    best_model = ""
    best_parameter = ""
    start_time = timeit.default_timer()

    for model_key in models:
        print("Starting " + model_key + " at " + str(datetime.datetime.now()))
        model = models[model_key]
        parameter_values = parameters_grid[model_key]
        
        for parameter in ParameterGrid(parameter_values):
            s = timeit.default_timer()
            model.set_params(**parameter)

            # Calculate AUC using 5-fold cross validation
            x_train_no_id = x_train.drop('id', axis=1)
            x_train_no_id_outcome = x_train_no_id.drop(outcome_label, axis=1)
            scores = cross_val_score(estimator=model,
                                     X=x_train_no_id_outcome,
                                     y=x_train[outcome_label],
                                     cv=5,
                                     scoring='roc_auc')
            auc = scores.mean()
            time = timeit.default_timer() - start_time
            results_df.loc[len(results_df)] = [model_key, parameter, auc, time]

            # Update "winner"
            if (auc > max_auc):
                max_auc = auc
                best_model = model
                best_parameter = parameter
                best_model_type = model_key

    elapsed = timeit.default_timer() - start_time

    print(results_df)
    print("Highest AUC " + str(max_auc))
    print("Best Model " + str(best_model))
    print("Best Parameter " + str(best_parameter))
    print('Total Time: ', elapsed)
    
    # Fit best model & best parameters on full training dataset
    best_model.set_params(**best_parameter)
    best_model.fit(x_train_no_id_outcome,
                   x_train[outcome_label])

    return best_model

def plot_precision_recall(y_test, y_hat, model, output_type='save'):
    '''
    Plot precision-recall curve for the test set.
    Input:
        y_test (array): true y values in test set
        y_hat (array): predicted y values for test set
        model (obj): fit training model (used for title)
        output_type (str): save or display image
    Return:
        Saved image.
    '''
    # Compute precision-recall pairs for different probability thresholds
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test,
                                                                          y_hat) 
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]

    # Transform scores thresholds percentage thresholds
    pct_above_per_thresh = []
    number_scored = len(y_hat)   

    for value in pr_thresholds:
        num_above_thresh = len(y_hat[y_hat >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    # Create plot
    plt.clf()
    fig, ax1 = plt.subplots()
    # Precision curve = blue
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('Percent of Images')
    ax1.set_ylabel('Precision', color='b')

    # Recall curve = red
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('Recall', color='r')
    
    ax1.set_ylim([0,1])
    ax1.set_xlim([0,1])
    ax2.set_ylim([0,1])
    
    # Name plot 
    model_name = str(model).split('(')[0]
    plot_name = model_name
    title = ax1.set_title(textwrap.fill(plot_name, 70))
    fig.tight_layout()
    fig.subplots_adjust(top=0.75)    

    #Save or show plot
    if (output_type == 'save'):
        plt.savefig(str(plot_name)+'.png')
    elif (output_type == 'show'):
        plt.show()
    plt.close()


def plot_auc(best_model, y_test, y_hat, output_type='save'):
  '''
  Plot and save ROC_AUC curve.
  '''
  # plt.clf()
  # fig, ax1 = plt.subplots()
  # plot_roc_curve(best_model, x_test, y_test)
  # title = ax1.set_title(textwrap.fill(plot_name, 70))
  # fig.tight_layout()
  # fig.subplots_adjust(top=0.75)  

  # Name plot 
  model_name = str(best_model).split('(')[0]
  plot_name = model_name
  # Find metrics
  fpr, tpr, thresholds = metrics.roc_curve(y_test, y_hat)
  roc_auc = metrics.auc(fpr, tpr)
  display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=plot_name) 
  display.plot()

  #Save or show plot
  if (output_type == 'save'):
      plt.savefig('ROC_'+ str(plot_name) +'.png')
  elif (output_type == 'show'):
      plt.show()
  plt.close()


def main(args):
    '''
    Full loop execution.
    Input:
        args (dict): dictionary of command line arguments
    Return:
        None
    '''
    models = {'Tree': DecisionTreeRegressor(max_depth=10),
              'Logistic': LogisticRegression(penalty='l1'),
              'Lasso': Lasso(alpha=0.1),
              'Ridge': Ridge(alpha=.5),
              'Forest': RandomForestRegressor(max_depth=2),
            #   'SVM': SVC(C=1, kernel='rbf'),
              'Bagging': BaggingClassifier(KNeighborsClassifier(), n_estimators=10),
              'AdaBoost': AdaBoostClassifier(n_estimators=50),
              'GradientBoost': GradientBoostingClassifier(learning_rate=0.05),
              }

    parameters_grid = {'Tree': {'max_depth': [10, 20, 50]},
                       'Logistic': {'penalty': ['l2'], 'C': [0.1, 1, 10]},
                       'Lasso': {'alpha': [0.01, 0.1, 1]},
                       'Ridge': {'alpha': [0.01, 0.1, 1]},
                       'Forest': {'max_depth': [10, 20, 50],
                                  'min_samples_split': [2, 10]},
                    #    'SVM': {'C': [1, 10],
                    #            'kernel': ['linear', 'rbf']}, #Takes the longest
                       'Bagging': {'n_estimators': [10, 100]},
                       'AdaBoost': {'algorithm': ['SAMME', 'SAMME.R'],
                                    'n_estimators': [10, 100]},
                       'GradientBoost': {'n_estimators': [10, 100]}
                       }

    outcome = 'pathology' # CHANGE THIS

    # CHANGE LOADING PROBABLY
    print("Loading features from csv...")
    train = pd.read_csv("current_train.csv")
    test = pd.read_csv("current_test.csv")

    best_model = find_best_model(models, parameters_grid, train, outcome)

    #Run predictions on test data and calculate AUC

    y_hats = best_model.predict(test.drop(columns=['id', 'pathology'],
                                            axis=1))
    roc_auc = roc_auc_score(test[outcome], y_hats)

    print("Testing Data ROC_AUC Score: ", roc_auc)
    
    # Generate Test Plots
    plot_precision_recall(test[outcome],
                            y_hats,
                            best_model,
                            'save')
    plot_auc(best_model,
                test[outcome],
                y_hats,
                'save')
        
    return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--train", default = "", help = "Training file path")
    parser.add_argument("-test", "--test", default= "", help = "Test file path")

    args = parser.parse_args()

    main(args)