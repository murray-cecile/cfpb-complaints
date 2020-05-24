import shallow_util as util 
from shallow_constants import * 

pd.set_option('expand_frame_repr', False)


def shallow_run(existing=True): 
    '''
    Runs through the full machine learning pipeline. This includes loading the raw data, 
    processing and creating features, running a grid search over model specifications, and  
    evaluating the best model. 

    The configuration is specified in the constants file.  
    
    Inputs: 
    - existing: boolean indicating whether to use existing resampled and processed data. 

    Outputs: 
    - Writes to a log file. Also stores the cross validation results and feature importance 
    from the best classifier. 
    '''

    if existing: 
        # Load data 
        print('\n ---------- LOADING DATA ---------- \n')
        X_train, X_test, y_train, y_test = util.load_existing_split() 

    else: 
        # Load data
        print('\n ---------- LOADING DATA ---------- \n')
        cfpb = util.load_data()
        cfpb = cfpb.sample(n=2000) 

        # Create features and label columns 
        cfpb_X = util.process_features(cfpb)
        cfpb_y = cfpb[RESPONSE_COL]

        # Split training and testing data 
        print('\n ---------- RESAMPLING TRAINING DATA ---------- \n')
        X_train, X_test, y_train, y_test = util.split_resample(cfpb_X, cfpb_y) 


    # Train models 
    print('\n ---------- TRAINING MODELS ---------- \n')
    best_clf = util.find_best_model(X_train, y_train)

    # Get predictions and probabilities 
    y_pred = best_clf.predict(X_test)
    y_pred_proba = best_clf.predict_proba(X_test)

    # Summarize predictions 
    print('\n ---------- EVALUATING PREDICTIONS ---------- \n')
    util.summarize_predictions(best_clf, y_test, y_pred)

    # Summarize probabilities 
    print('\n ---------- EVALUATING PROBABILITIES ---------- \n')
    util.summarize_probabilities(best_clf, y_test, y_pred_proba)

    # Summarize feature importance 
    print('\n ---------- FEATURE IMPORTANCE ---------- \n')
    feature_importance = util.feature_importance(best_clf, X_train)


if __name__ == "__main__":
    
    shallow_run() 





