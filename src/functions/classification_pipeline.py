import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_predict
import os

from settings import Settings
settings = Settings()

def classification_pipeline(x_train, y_train, pipe, cv, scoring, param_grid):

    cv_ = cv
    scoring_ = scoring
    param_grid_ = param_grid

    reg = GridSearchCV(pipe, cv = cv_, scoring=scoring_, param_grid=param_grid_)
    reg.fit(x_train, y_train)

    # get the explained variance achieved with dimension reduction -> see if # of dimensions can be reduced further
    #reg.best_estimator_.named_steps['reduce_dim'].explained_variance_


    # get scores for regression model
    means = reg.cv_results_['mean_test_score']
    stds = reg.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, reg.cv_results_['params']):
        print(str(reg.scoring) + " Score: " +  "%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print('Best ' + str(reg.scoring) + " Score: " + str(max(means)))
    pipe_best_params = reg.best_params_

    # get best parameter to fit them into the testing model
    return pipe_best_params


def evaluate_pipe_best_train(x_train, y_train, pipe_best, algo):
    pipe_best.fit(x_train, y_train)
    y_train_predict = pipe_best.predict(x_train)
    print(classification_report(y_train, y_train_predict))
    print('Accuracy Train: {}'.format(accuracy_score(y_train, y_train_predict)))


#    def plot_confusion_matrix():


    if algo in ("RandomForestClassifier", "KNeighborsClassifier"):
        y_train_proba = cross_val_predict(pipe_best, x_train, y_train, cv=5, method="predict_proba")
        y_train_scores = y_train_proba[:, 1]

    else:
        y_train_scores = cross_val_predict(pipe_best, x_train, y_train, cv=5, method="decision_function")


    def plot_precision_recall_vs_threshold(y_train, y_train_scores, algo, output_file_path=None):
        precicions, recalls, thresholds = precision_recall_curve(y_train, y_train_scores)
        fig10 = plt.figure(10)
        plt.plot(thresholds, precicions[:-1], 'b--', label='Precision')
        plt.plot(thresholds, recalls[:-1], 'g--', label='Recall')
        plt.xlabel('Threshold')
        plt.legend(loc='upper left')
        plt.ylim([0,1])
        if output_file_path is not None:
            fig10.savefig(os.path.join(output_file_path, 'precision_recall_vs_threshold_' + str(algo) + '.pdf'))

    plot_precision_recall_vs_threshold(y_train, y_train_scores, algo, output_file_path=settings.figures)

    def plot_roc_curve(y_train, y_train_scores, algo, output_file_path=None):
        fpr, tpr, thresholds = roc_curve(y_train, y_train_scores)
        fig11 = plt.figure(11)
        plt.plot(fpr, tpr, linewidth=2, label=None)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if output_file_path is not None:
            fig11.savefig(os.path.join(output_file_path, 'roc_curve_' + str(algo) + '.pdf'))

    plot_roc_curve(y_train, y_train_scores, algo, output_file_path=settings.figures)



def evaluate_pipe_best_test(x_test, y_test, pipe_best):
    y_test_predict = pipe_best.predict(x_test)
    print(classification_report(y_test, y_test_predict))
    print('Accuracy Test: {}'.format(accuracy_score(y_test, y_test_predict)))