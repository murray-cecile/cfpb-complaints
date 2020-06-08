import shallow_util as util 
from shallow_constants import * 

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


pd.set_option('expand_frame_repr', False)


def shallow_run(): 
    '''
    Runs through the full machine learning pipeline. This includes loading the raw data, 
    processing the text features, running a grid search over model specifications, and  
    evaluating theese models. 

    The configuration is specified in the constants file.  
    
    Outputs: 
    - Writes to a log file. 
    '''

    # Load data 
    print('\n ---------- LOADING DATE ---------- \n')
    X_train, X_test, y_train, y_test = util.load_data_task2()

    # NAIVE BAYES 
    print('\n ---------- NAIVE BAYES---------- \n')

    # Create Naive Bayes pipeline 
    text_clf_nb = Pipeline(
        [('vect', CountVectorizer()), 
        ('tfidf', TfidfTransformer()), 
        ('clf', MultinomialNB())])

    # Create grid search object 
    nb_clf = GridSearchCV(estimator=text_clf_nb, 
                      param_grid=PARAMS_NB, 
                      cv=5)

    # Fit and predict pipeline
    nb_clf = nb_clf.fit(X_train, y_train)

    print(nb_clf.best_score_)
    print(nb_clf.best_params_)

    nb_predictions = nb_clf.predict(X_test) 
    print(classification_report(y_test, nb_predictions))

    
    # SVM 
    print('\n ---------- SVM---------- \n')
    
    # Create Naive Bayes pipeline 
    text_clf_svm = Pipeline(
        [('vect', CountVectorizer()), 
         ('tfidf', TfidfTransformer()),
         ('clf', SGDClassifier(loss='hinge', penalty='l2'))])

    # Create grid search object 
    svm_clf = GridSearchCV(estimator=text_clf_svm, 
                       param_grid=PARAMS_SVM, 
                       cv=5)

   # Fit and predict pipeline
    svm_clf = svm_clf.fit(X_train, y_train)

    print(svm_clf.best_score_)
    print(svm_clf.best_params_)

    svm_predictions = svm_clf.predict(X_test)
    print(classification_report(y_test, svm_predictions))

    print('\n -------------------- \n')


if __name__ == "__main__":
    
    shallow_run() 





