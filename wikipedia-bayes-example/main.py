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
