# Shallow learner classifiers 

This subdirectory contains the pipeline to run a run a series of One-vs-Rest classifiers for the first prediction task (predicting the company response). Note that the pipeline (even with a small grid of hyperparamters takes several days to run even on a compute-optimized EC2 instance). 

- `shallow_constants.py` contains the model setup configuration 

- `shallow_main.py` runs the full machine learning pipeline: 
	- Loads and pre-processes the raw data 
	- Engineers the relevant features 
	- Splits into training and testing sets, and then resamples the training data 
	- Uses k-fold cross-validation to iterate over various classifiers and hyperparameters 
	- Evaluates the best performing classifier 

- `shallow_util.py` contains the set of helper functions used in `shallow_main.py` 

- `shallow-evaluation.ipynb` contains post-modeling analysis (primarily for the final presentation) 

- `processed_data` contains processed and resampled modeling datasets in CSV format (the feature engineering and resampling take awhile, so I run this once and stored these files locally) 

- `results` contains the output of running the full pipeline 
	- `model_evaluation.csv` summarizes the best performing specification for each type of classifier 
	- `model_evaluation_full.csv` summarizes all specifications 
	- `shallow_log.out` logs the pipeline run 
	- `feature_importance.json` stores the feature importance for the best performing model for each label class 


### How to run models

Users should only need to modify `shallow_constants.py` to run various pipeline configurations. All configuration options (e.g. file paths, feature engineering specifications, cross-validation parametes, etc.) are managed in this file. 

To run the full machine learning pipeline from the command line: 

```
python3 shallow_main.py > results/shallow_log.out 
```
