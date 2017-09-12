from Orange.data import Domain, Table
from Orange.classification import LogisticRegressionLearner
from Orange.classification import NaiveBayesLearner
from Orange.classification import TreeLearner
from Orange.classification import RandomForestLearner
from Orange.classification import KNNLearner
from Orange.classification import SVMLearner

### create models ###

models = [
    LogisticRegressionLearner(),
    NaiveBayesLearner(),
    TreeLearner(),
    RandomForestLearner(), 
    KNNLearner(),
    SVMLearner(),
]
        
### read train data ###
        
train = Table.from_file('train.csv')
# move `sex` from X to Y (from attributes/features to class_var/target)
domain = Domain(train.domain.attributes[1:], train.domain.attributes[0])
train = train.transform(domain)

print('\n=== train.X ===')
print(train.X)
print('\n=== train.Y ===')
print(train.Y)

### read predict data ###

predict = Table.from_file('predict.csv')

print('\n=== predict.X ===')
print(predict.X)

### learn and predict ###

print('\n=== results ===')

class_values = train.domain.class_var.values

for learner in models:
    # train
    classifier = learner(train)
    
    # predict
    result = classifier(predict)
    
    # print result
    result = class_values[int(result[0])]
    print('{:25s}: {}'.format(learner.__class__.__name__, result))
    
