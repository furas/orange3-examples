import pandas as pd
#from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.svm import SVC

import numpy as np
from mlfromscratch.supervised_learning import Adaboost
from mlfromscratch.supervised_learning import ClassificationTree
from mlfromscratch.supervised_learning import KNN
from mlfromscratch.supervised_learning import LogisticRegression
from mlfromscratch.supervised_learning import NaiveBayes
from mlfromscratch.supervised_learning import RandomForest
from mlfromscratch.supervised_learning import SupportVectorMachine
from mlfromscratch.supervised_learning import XGBoost

### create models ###

models = [
    #Adaboost(), # error: `float` has no `exp`
    ClassificationTree(),
#    KNN(), # doesn't have fit(), it uses predict(test_X, train_X, train_Y)
#    LogisticRegression(), # not work
    NaiveBayes(),
    RandomForest(), 
#    SupportVectorMachine(),
    XGBoost(),
]

### read train data ###

train_X = pd.read_csv('train.csv')
# move `sex` from X to Y
train_Y = train_X.pop('sex')

# get numpy array
train_X = train_X.values
train_Y = train_Y.values

# convert names (female/male) into values (0/1)
# because model can't work with strings
target_names = np.unique(train_Y)
for val, name in enumerate(target_names):
    train_Y[train_Y == name] = val

print('\n=== train_X ===')
print(train_X)
print('\n=== train_Y ===')
print(train_Y)

### read predict data ###

predict_X = pd.read_csv('predict.csv')
predict_X = predict_X.values

print('\n=== predict_X ===')
print(predict_X)

### learn and predict ###

print('\n=== results ===')

for m in models:
    # train
    m.fit(train_X, train_Y)
    
    # predict
    result = m.predict(predict_X)
    
    # print result
    result = target_names[result[0]]
    print('{:25s}: {}'.format(m.__class__.__name__, result))

#----------------------------------------------------------------------

m = KNN()

# predict
result = m.predict(predict_X, train_X, train_Y)

# print result
result = target_names[result[0]]
print('{:25s}: {}'.format(m.__class__.__name__, result))

#----------------------------------------------------------------------
