
import load_data_set as ld
from create_histogram_feature import get_feature

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import f1_score, make_scorer


def normalize(X_tr, X_te):
    ''' Normalize training and test data features
    Args:
        X_tr: Unnormalized training features
        X_te: Unnormalized test features
    Output:
        X_tr: Normalized training features
        X_te: Normalized test features
    '''
    X_mu = np.mean(X_tr, axis=0)
    X_tr = X_tr - X_mu
    X_sig = np.std(X_tr, axis=0)
    X_tr = X_tr/X_sig
    X_te = (X_te - X_mu)/X_sig
    return X_tr, X_te

def score_f1_class1(ground_truth, predictions):
    '''
    Returns f1 score for class 1
    '''
    return f1_score(ground_truth, predictions, average='binary', pos_label=1)

X_tr, X_te = [],[]

for train in ld.X_train :
    X_tr.append(get_feature(train))
    
for test in ld.X_test:
    X_te.append(get_feature(test))

X_tr, X_te = np.array(X_tr), np.array(X_te)

col_deleted = np.nonzero((X_tr==0).sum(axis=0) > (len(X_tr)-1000))
# col_deleted = col_deleted[0].tolist() + range(6,22) + range(28,44)
print col_deleted
X_tr = np.delete(X_tr, col_deleted, axis=1)
X_te = np.delete(X_te, col_deleted, axis=1)


X_tr, X_te = normalize(X_tr, X_te)
y_tr, y_te = ld.y_train, ld.y_test

print "training and testing data normalised ", len(y_tr), len(y_te)


gamma_ramge = [ 4**i for i in range(-6,-2) ]
C_range = [ 4**i for i in range(2,3) ] #-1,6

score = make_scorer(score_f1_class1, greater_is_better=True)

# Number of folds in Cross validation
CV_FOLDS = 3
# Number of parallel jobs
parallel = -1

svr = SVC()

parameters = [{ 'kernel':['rbf'], 'C':C_range, 'gamma':gamma_ramge}]#, 'tol':[1e-4], 'max_iter':[50000]}]

clf = GridSearchCV(svr, parameters, cv=CV_FOLDS, n_jobs = parallel, verbose=1000, iid=False)# , scoring=score

clf.fit(X_tr,y_tr)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
mean_fit_time = clf.cv_results_['mean_fit_time']

for mean, std, params,time in zip(means, stds, clf.cv_results_['params'],mean_fit_time):
    print("%0.3f (+/-%0.03f) for %r time-%0.3f"
          % (mean, std * 2, params,time))
print ""

print "Detailed classification report:"
print
y_true, y_pred = y_te, clf.predict(X_te)
print(classification_report(y_true, y_pred))
print ""

print "done"    
    
