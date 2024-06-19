import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


import numpy as np

# Unpickling the dataset
with open('newdata.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['label'])



# Now we will split into training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels,random_state=104 ,test_size=0.25, shuffle=True, stratify=labels)




# for RandomForrest
from sklearn.ensemble import RandomForestClassifier

model1 = RandomForestClassifier()
model1.fit(x_train, y_train)  # Training our model
y_predict1 = model1.predict(x_test)  # Testing our model
#train_score1 = model1.score(x_train, y_train)  # Training Accuracy
test_score1 = accuracy_score(y_test, y_predict1)  # Testing Accuracy
print("Using RandomForestClassifier")
print("{}% of the samples were classified correctly! [ Testing Accuracy ]".format(test_score1 * 100))
print(classification_report(y_test, y_predict1))

print("-------------------------------------------------------------------------------------------------------")

# for KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier(n_neighbors=7)
model2.fit(x_train, y_train)  
y_predict2 = model2.predict(x_test)  # Testing our model
#train_score2 = model2.score(x_train, y_train)  # Training Accuracy
test_score2 = accuracy_score(y_test, y_predict2)  # Testing Accuracy
print("Using KNeighborsClassifier")
print("{}% of the samples were classified correctly! [ Testing Accuracy ]".format(test_score2 * 100))
print(classification_report(y_test, y_predict2))


print("-------------------------------------------------------------------------------------------------------")

# for DecisionTreeClassifier
from sklearn import tree

model3 = tree.DecisionTreeClassifier()
model3.fit(x_train, y_train)  
y_predict3 = model3.predict(x_test)  # Testing our model
#train_score3 = model3.score(x_train, y_train)  # Training Accuracy
test_score3 = accuracy_score(y_test, y_predict2)  # Testing Accuracy
print("Using DecisionTreeClassifier")
print("{}% of the samples were classified correctly! [ Testing Accuracy ]".format(test_score3 * 100))
print(classification_report(y_test, y_predict3))




# Saving/Pickling our model
with open('model1.p', 'wb') as f:
    pickle.dump({'model': model1}, f)
f.close()
