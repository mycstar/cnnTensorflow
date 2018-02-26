from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn import svm, datasets

# A sample toy binary classification dataset
X, y = datasets.make_classification(n_classes=2, random_state=0)
svm = svm.LinearSVC(random_state=0, verbose=1)


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]


def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


scoring = {'tp' : make_scorer(tp), 'tn' : make_scorer(tn), 'fp' : make_scorer(fp), 'fn' : make_scorer(fn)}

cv_results = cross_validate(svm.fit(X, y), X, y,  verbose=1)

# Getting the test set true positive scores
print(cv_results['test_tp'])

# Getting the test set false negative scores
print(cv_results['test_fn'])