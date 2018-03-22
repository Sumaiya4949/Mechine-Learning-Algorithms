
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('iris.csv')
X = data.values[:, :4]
y = np.zeros(150)
for i in range(len(y)):
    if data.values[i, 4]=='setosa':
        y[i] = 0
    elif data.values[i, 4]=='versicolor':
        y[i] = 1
    elif data.values[i, 4]=='virginica':
        y[i] = 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

KNN_model = KNeighborsClassifier(n_neighbors=3)
KNN_model.fit(X_train, y_train)
y_predict = KNN_model.predict(X_test)
accuracy_score(y_test, y_predict)

