# Data

Example data from:
[Wikipedia > Naive_Bayes_classifier > Examples](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Examples)


__Train data (train.csv)__

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


__Prediction data (test.csv)__

     sex    | height (feet) | weight (lbs) | foot size(inches)
    --------+---------------+--------------+-----------------
     ?      | 6             | 130          | 8

__Prediction result__: 
    
    female

---

# Orange3 

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

### Data Table (test)
![](images/data-table-test.png?raw=true)

### Predictions
![](images/predictions.png?raw=true)

### Data Table (result)
![](images/data-table-result.png?raw=true)

---

# SKLearn 

### main.py

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

    === train_y ===
    0      male
    1      male
    2      male
    3      male
    4    female
    5    female
    6    female
    7    female
    Name: sex, dtype: object

    === test_X ===
       height  weight  foot_size
    0       6     130          8

    === results ===
    GaussianNB             : female
    LogisticRegression     : female
    DecisionTreeClassifier : female
    KNeighborsClassifier   : female
    RandomForestClassifier : female
    AdaBoostClassifier     : female
    SVC                    : female

