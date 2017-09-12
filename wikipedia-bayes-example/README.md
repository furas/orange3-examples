Table of contents
=================

* [Data](#data)
    * [Train data (train.csv)](#train-data-traincsv)
    * [Prediction data (predict.csv)](#prediction-data-predictcsv)
    * [Prediction result:](#prediction-result:)
* [Using GUI ([Orange3](https://orange.biolab.si/))](#using-gui-[orange3]https://orangebiolabsi/)
    * [wikipedia-bayes-example.ows](#wikipedia-bayes-exampleows)
    * [Data Table (train)](#data-table-train)
    * [Test & Score](#test-&-score)
    * [Confusion Matrix](#confusion-matrix)
    * [Tree Viewer](#tree-viewer)
    * [Data Table (predict)](#data-table-predict)
    * [Predictions](#predictions)
    * [Data Table (result)](#data-table-result)
* [Using code](#using-code)
    * [orange-version.py](#orange-versionpy)
    * [sklearn-version.py](#sklearn-versionpy)
    * [ml-from-scratch-version.py](#ml-from-scratch-versionpy)

---
        
# Data

Example data from:
[Wikipedia > Naive_Bayes_classifier > Examples](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Examples)


### Train data (train.csv)

     sex    | height (feet) | weight (lbs) | foot size(inches)
    --------+---------------+--------------+-----------------
     male   | 6             | 180          | 12
     male   | 5.92 (5'11")  | 190          | 11
     male   | 5.58 (5'7")   | 170          | 12
     male   | 5.92 (5'11")  | 165          | 10
     female | 5             | 100          | 6
     female | 5.5 (5'6")    | 150          | 8
     female | 5.42 (5'5")   | 130          | 7
     female | 5.75 (5'9")   | 150          | 9


### Prediction data (predict.csv)

     sex    | height (feet) | weight (lbs) | foot size(inches)
    --------+---------------+--------------+-----------------
     ?      | 6             | 130          | 8

### Prediction result: 
    
    female

---

# Using GUI ([Orange3](https://orange.biolab.si/))

### wikipedia-bayes-example.ows
![](images/wikipedia-bayes-example.png?raw=true)

### Data Table (train)
![](images/data-table-train.png?raw=true)

### Test & Score
![](images/test-and_score.png?raw=true)

### Confusion Matrix
![](images/confusion-matrix.png?raw=true)

### Tree Viewer
![](images/tree-viewer.png?raw=true)

### Data Table (predict)
![](images/data-table-predict.png?raw=true)

### Predictions
![](images/predictions.png?raw=true)

### Data Table (result)
![](images/data-table-result.png?raw=true)

---

# Using code

### orange-version.py

    === train.X ===
    [[   6.    180.     12.  ]
     [   5.92  190.     11.  ]
     [   5.58  170.     12.  ]
     [   5.92  165.     10.  ]
     [   5.    100.      6.  ]
     [   5.5   150.      8.  ]
     [   5.42  130.      7.  ]
     [   5.75  150.      9.  ]]
    
    === train.Y ===
    [ 1.  1.  1.  1.  0.  0.  0.  0.]
    
    === predict.X ===
    [[   6.  130.    8.]]
    
    === results ===
    LogisticRegressionLearner: female
    NaiveBayesLearner        : female
    TreeLearner              : female
    RandomForestLearner      : female
    KNNLearner               : female
    SVMLearner               : female

### sklearn-version.py

[scikit-learn](http://scikit-learn.org/stable/) + [pandas](http://pandas.pydata.org/)

    === train_X ===
       height  weight  foot_size
    0    6.00     180         12
    1    5.92     190         11
    2    5.58     170         12
    3    5.92     165         10
    4    5.00     100          6
    5    5.50     150          8
    6    5.42     130          7
    7    5.75     150          9
    
    === train_Y ===
    0      male
    1      male
    2      male
    3      male
    4    female
    5    female
    6    female
    7    female
    Name: sex, dtype: object
    
    === predict_X ===
       height  weight  foot_size
    0       6     130          8
    
    === results ===
    GaussianNB               : female
    LogisticRegression       : female
    DecisionTreeClassifier   : female
    KNeighborsClassifier     : female
    RandomForestClassifier   : female
    AdaBoostClassifier       : female
    SVC                      : female

###ml-from-scratch-version.py
  
[ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
    
    === train_X ===
    [[   6.    180.     12.  ]
     [   5.92  190.     11.  ]
     [   5.58  170.     12.  ]
     [   5.92  165.     10.  ]
     [   5.    100.      6.  ]
     [   5.5   150.      8.  ]
     [   5.42  130.      7.  ]
     [   5.75  150.      9.  ]]
    
    === train_Y ===
    [1 1 1 1 0 0 0 0]
    
    === predict_X ===
    [[  6 130   8]]
    
    === results ===
    ClassificationTree       : female
    NaiveBayes               : female
    Training: 100% [-----------------------------------------------------------------------------------------] Time: 0:00:00
    RandomForest             : female
    Training: 100% [-----------------------------------------------------------------------------------------] Time: 0:00:01
    XGBoost                  : female
    KNN                      : female
    

