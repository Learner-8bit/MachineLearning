#import sys
import os
import time
from sklearn import metrics
import numpy as np
import pickle
import importlib,sys
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#from numpy.core.umath_tests import inner1d

importlib.reload(sys)


#sys.setdefaultencoding('utf8')


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model





def K(X, Y=None, metric='poly', coef0=1, gamma=None, degree=3):
    if metric == 'poly':
        k = pairwise_kernels(X, Y=Y, metric=metric, coef0=coef0, gamma=gamma, degree=degree)
    elif metric == 'linear':
        k = pairwise_kernels(X, Y=Y, metric=metric)
    elif metric == 'sigmoid':
        k = pairwise_kernels(X, Y=Y, metric=metric, coef0=coef0, gamma=gamma)
    elif metric == 'rbf':
        k = pairwise_kernels(X, Y=Y, metric=metric, gamma=gamma)
    return k


class KernelKNN():
    def __init__(self, metric='poly', coef0=1, gamma=None, degree=3,
                 n_neighbors=5, weights='distance'):  # uniform
        self.m = metric
        self.coef = coef0
        self.gamma = gamma
        self.degree = degree
        self.n = n_neighbors
        self.weights = weights

    def fit(self, X, Y=None):
        # Polynomial kernel: K(a,b) = (coef0+gamma<a,b>)**degree
        # Sigmoid kernel: K(a,b) = tanh(coef+gamma<a,b>)
        # Linear kernel: K(a,b) = <a,b>
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.x = X
        self.y = Y
        classes = np.unique(Y)
        label = np.zeros((len(Y), len(classes)))
        for i in range(len(Y)):
            for ii in range(len(classes)):
                if Y[i] == classes[ii]:
                    label[i][ii] = 1
        self.classes = classes
        self.label = label

        Kaa = []
        for i in range(len(X)):
            Kaa.append(K(X[i, :].reshape(1, -1), metric=self.m,
                         coef0=self.coef, gamma=self.gamma, degree=self.degree))
        self.Kaa = np.asarray(Kaa).ravel().reshape(len(Kaa), 1)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        Kbb = []
        Kab = K(self.x, X, metric=self.m,
                coef0=self.coef, gamma=self.gamma, degree=self.degree)

        for i in range(len(X)):
            Kbb.append(K(X[i, :].reshape(1, -1), metric=self.m,
                         coef0=self.coef, gamma=self.gamma, degree=self.degree))
        self.Kbb = np.asarray(Kbb).ravel()
        d = np.array(self.Kaa - 2 * Kab + self.Kbb, dtype=float)  # shape: (n_train,n_test)

        n_d = []  # neighbors' distance matrix
        index = []
        for i in range(d.shape[1]):
            index.append(np.argsort(d[:, i])[:self.n])
            n_d.append(d[index[i], i])
        n_d = np.asmatrix(n_d) + 1e-20

        w = np.asarray((1 / n_d) / np.sum(1 / n_d, axis=1))
        # weights matrix, shape: (n_test,n_neighbors)
        w_neighbor = w.reshape((w.shape[0], 1, w.shape[1]))
        # neighbors' weights matrix, shape: (n_test,1,n_neighbors)

        prob = []
        label_neighbor = self.label[index]
        # neighbors' index, shape: (n_test,n_neighbors,n_classes)
        for i in range(len(w_neighbor)):
            prob.append(np.dot(w_neighbor[i, :, :], label_neighbor[i, :, :]).ravel())
        prob = np.asarray(prob)
        self.prob = prob
        return prob

    def predict(self):

        # prob = predict_proba(self,X)
        yhat = self.classes[np.argmax(self.prob, axis=1)]
        return yhat

def read_data(data_file):
    import gzip
    f = gzip.open(data_file, "rb")
    train, val, test = pickle.load(f,encoding='bytes')
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    data_file = "mnist.pkl.gz"
    thresh = 0.5
    model_save_file = None
    model_save = {}

    test_classifiers = ['NB', 'KNN', 'LR', 'SVM']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'SVM': svm_classifier,
                   }

    print('reading training and testing data...')
    train_x, train_y, test_x, test_y = read_data(data_file)
    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape
    is_binary_class = (len(np.unique(train_y)) == 2)
    print('******************** Data Info *********************')
    print('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))

    for classifier in test_classifiers:
        print('\n******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))
        
        start_time = time.time()
        
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        if is_binary_class:
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        
        print('testing took %fs!' % (time.time() - start_time))
        
        print('accuracy: %.2f%%' % (100 * accuracy))

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))


    acc=[]
    acc_p=[]
    acc_l=[]
    acc_r=[]
    acc_s=[]

    for k in range(1, 10):
        KNN = KNeighborsClassifier(n_neighbors=k, p=2, n_jobs=-1)
        KNN.fit(train_x, train_y)
        y_pred=KNN.predict(test_x)
        acc.append(accuracy_score(test_y, y_pred))
        KNN_p = KernelKNN(n_neighbors=k, metric='poly', gamma=1/784)
        KNN_p.fit(train_x, train_y)
        KNN_p.predict_proba(test_x)
        y_pred = KNN_p.predict()
        acc_p.append(accuracy_score(test_y, y_pred))
        KNN_l = KernelKNN(n_neighbors=k, metric='linear', gamma=1/784)
        KNN_l.fit(train_x, train_y)
        KNN_l.predict_proba(test_x)
        y_pred = KNN_l.predict()
        acc_l.append(accuracy_score(test_y, y_pred))
        #KNN_r = KernelKNN(n_neighbors=k, metric='rbf', gamma=1/784)
        #KNN_r.fit(train_x, train_y)
        #KNN_r.predict_proba(test_x)
        #y_pred = KNN_r.predict()
        #acc_r.append(accuracy_score(test_y, y_pred))
        KNN_s = KernelKNN(n_neighbors=k, metric='sigmoid', gamma=1/784)
        KNN_s.fit(train_x, train_y)
        KNN_s.predict_proba(test_x)
        y_pred = KNN_s.predict()
        acc_s.append(accuracy_score(test_y, y_pred))



    plt.xlabel('k')
    plt.ylabel('Acuuracy')
    plt.title('Accuracy Curves of Kelnels')
    plt.plot(acc, label='none')
    plt.plot(acc_p, label='poly')
    plt.plot(acc_s, label='sigmoid')
    #plt.plot(acc_r, label='rbf')
    plt.plot(acc_l, label='linear')
    plt.legend()
