import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

### create models ###

models = [
    GaussianNB(),
    LogisticRegression(),
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    RandomForestClassifier(), 
    AdaBoostClassifier(),
    SVC(),
]
        
### read train data ###
        
train_X = pd.read_csv('train.csv')
# move `sex` from X to Y
train_Y = train_X.pop('sex')

print('\n=== train_X ===')
print(train_X)
print('\n=== train_Y ===')
print(train_Y)

### read predict data ###

predict_X = pd.read_csv('predict.csv')

print('\n=== predict_X ===')
print(predict_X)

### train and predict ###

print('\n=== results ===')

for m in models:
    # train
    m.fit(train_X, train_Y)
    
    # predict
    result = m.predict(predict_X)
    
    # print result
    result = result[0]
    print('{:25s}: {}'.format(m.__class__.__name__, result))
