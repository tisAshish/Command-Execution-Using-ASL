# Command Execution Using ASL

## Data Sample
![ASL](https://github.com/tisAshish/Command-Execution-Using-ASL/assets/88030649/1e40a0fa-9f08-41ed-b0dc-379fd2e3853a)

## Lazy Predict
```python
import lazypredict
from lazypredict.Supervised import LazyClassifier
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

```


```python

# Unpickling the dataset
with open('newdata.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['label'])
```


```python
X_train, X_test, y_train, y_test = train_test_split(data, labels,random_state=104 ,test_size=0.25, shuffle=True, stratify=labels)

clf = LazyClassifier()
train,test = clf.fit(X_train, X_test, y_train, y_test)

print(train)
```

     90%|█████████████████████████████████████████████████████████▍      | 26/29 [00:24<00:02,  1.45it/s]

    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001516 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 10710
    [LightGBM] [Info] Number of data points in the train set: 5597, number of used features: 42
    

    100%|████████████████████████████████████████████████████████████████| 29/29 [00:28<00:00,  1.01it/s]

                                   Accuracy  Balanced Accuracy ROC AUC  F1 Score  \
    Model                                                                          
    QuadraticDiscriminantAnalysis      0.99               0.99    None      0.99   
    LogisticRegression                 0.99               0.99    None      0.99   
    CalibratedClassifierCV             0.99               0.99    None      0.99   
    LinearSVC                          0.99               0.99    None      0.99   
    LinearDiscriminantAnalysis         0.99               0.99    None      0.99   
    SVC                                0.98               0.98    None      0.98   
    PassiveAggressiveClassifier        0.98               0.98    None      0.98   
    RidgeClassifierCV                  0.98               0.98    None      0.98   
    RidgeClassifier                    0.98               0.98    None      0.98   
    SGDClassifier                      0.97               0.97    None      0.97   
    LGBMClassifier                     0.97               0.97    None      0.97   
    NuSVC                              0.97               0.97    None      0.97   
    ExtraTreesClassifier               0.97               0.97    None      0.96   
    RandomForestClassifier             0.96               0.96    None      0.96   
    Perceptron                         0.96               0.96    None      0.96   
    LabelSpreading                     0.95               0.95    None      0.95   
    LabelPropagation                   0.95               0.95    None      0.95   
    BaggingClassifier                  0.94               0.95    None      0.94   
    KNeighborsClassifier               0.93               0.93    None      0.93   
    DecisionTreeClassifier             0.89               0.89    None      0.89   
    ExtraTreeClassifier                0.79               0.79    None      0.79   
    GaussianNB                         0.72               0.73    None      0.72   
    NearestCentroid                    0.68               0.68    None      0.68   
    BernoulliNB                        0.61               0.61    None      0.60   
    AdaBoostClassifier                 0.27               0.26    None      0.17   
    DummyClassifier                    0.04               0.04    None      0.00   
    
                                   Time Taken  
    Model                                      
    QuadraticDiscriminantAnalysis        0.11  
    LogisticRegression                   0.57  
    CalibratedClassifierCV               2.75  
    LinearSVC                            0.53  
    LinearDiscriminantAnalysis           0.08  
    SVC                                  0.88  
    PassiveAggressiveClassifier          0.21  
    RidgeClassifierCV                    0.07  
    RidgeClassifier                      0.05  
    SGDClassifier                        0.27  
    LGBMClassifier                       3.60  
    NuSVC                                3.07  
    ExtraTreesClassifier                 0.77  
    RandomForestClassifier               2.91  
    Perceptron                           0.18  
    LabelSpreading                       4.22  
    LabelPropagation                     3.16  
    BaggingClassifier                    1.87  
    KNeighborsClassifier                 0.20  
    DecisionTreeClassifier               0.33  
    ExtraTreeClassifier                  0.03  
    GaussianNB                           0.04  
    NearestCentroid                      0.03  
    BernoulliNB                          0.04  
    AdaBoostClassifier                   2.47  
    DummyClassifier                      0.02  
    
