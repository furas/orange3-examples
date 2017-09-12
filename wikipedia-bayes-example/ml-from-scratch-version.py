import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

''' create models '''

models = [
    GaussianNB(),
    LogisticRegression(),
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    RandomForestClassifier(), 
    AdaBoostClassifier(),
    SVC(),
]
        
''' read train data '''
        
train_X = pd.read_csv('train.csv')
train_y = train_X.pop('sex')

print('\n=== train_X ===')
print(train_X)
print('\n=== train_y ===')
print(train_y)

''' read test data '''

test_X = pd.read_csv('test.csv')

print('\n=== test_X ===')
print(test_X)

''' learn and predict '''

print('\n=== results ===')

for m in models:
    # learn
    m.fit(train_X, train_y)
    
    # predict
    result = m.predict(test_X)
    
    # print result
    print('{:23s}: {}'.format(m.__class__.__name__, result[0]))



import mlfromscratch.supervised_learning
from mlfromscratch.supervised_learning.neural_network import NeuralNetwork
from mlfromscratch.utils.optimizers import GradientDescent
from mlfromscratch.utils.loss_functions import SquareLoss
from mlfromscratch.utils.layers import Dense, Activation
from mlfromscratch.utils.data_operation import accuracy_score

import numpy as np

X_train = np.array([[0,0], [0,1], [1,0], [1, 1]])
y_train = np.array([[1], [0], [0], [1]])
#y_train = np.array([1, 0, 0, 1])

clf = NeuralNetwork(optimizer=GradientDescent,
                    loss=SquareLoss)

clf.add(Dense(n_units=3, input_shape=(2,)))
clf.add(Activation('sigmoid'))
clf.add(Dense(n_units=1))#, input_shape=(2,)))
clf.add(Activation('sigmoid'))
print(y_train)

train_err, val_err = clf.fit(X=X_train, y=y_train, n_epochs=150, batch_size=4)
print('train_err:', train_err)
print('val_err:', val_err)

y_pred = np.argmax(clf.predict(X_train), axis=1)
print('pred:', y_pred)

accuracy = accuracy_score(y_train, y_pred)
print('Accuracy:', accuracy)

#print(clf.summary())

for x in clf.layers:
    print(x.layer_input)
    print(dir(x))
    print(dir(x.parameters))
    print(dir(x.activation))
    print(dir(x.activation.function))
    print(dir(x.activation.gradient))
    print(x.activation.gradient.im_class)
    print(dir(x.activation.gradient.im_func))
    print(x.activation.gradient.im_func.func_closure)
    print(x.activation.gradient.im_func.func_code)
    print(x.activation.gradient.im_func.func_defaults)
    print(x.activation.gradient.im_func.func_dict)
    #print(x.activation.gradient.im_func.func_doc)
    #rint(x.activation.gradient.im_func.func_globals)
    print(x.activation.gradient.im_func.func_name)
['func_closure', 'func_code', 'func_defaults', 'func_dict', 'func_doc', 'func_globals', 'func_name']
