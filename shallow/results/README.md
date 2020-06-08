Note that in order to get the models to run in a reasonable amount of time, we had to run a smaller set of hyperpamater options, and we had to parallelize this pipeline by running each classifier on a separate EC2 instance. Even with this parallelized approach, the process takes several days to run. As a result, the output files reflect the pipeline output on a development dataset (to demonstrate the full pipeline process). The results manually collated from these instances are in `shallow_log_FULL.out`. 


- `shallow_log_task1.out` logs the pipeline run for the first machine learning task 
- `model_evaluation_task1.csv` summarizes the best performing specification for each type of classifier 
- `model_evaluation_full_task1.csv` summarizes all specifications 
- `feature_importance_task1.json` stores the feature importance for the best performing model for each label class 

- `shallow_log_task2.out` logs the pipeline run for the second machine learning task 
