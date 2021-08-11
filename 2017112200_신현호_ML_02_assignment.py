#!/usr/bin/env python
# coding: utf-8

# In[24]:


import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sklearn
#0
a_list = pd.read_csv('C:\\신현호\\머신러닝\\ionosphere.csv')
a_list


# In[25]:


#1
a_list = a_list.dropna()
le = LabelEncoder()
a_list['class'] = le.fit_transform(a_list['class'])
a_list_enc = a_list
a_list_enc


# In[26]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#2-1
min_max_scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = min_max_scaler.fit_transform(a_list_enc.iloc[:,:-1])
a_list_enc_norm = pd.DataFrame(data_scaled)
class_column = a_list_enc['class']
a_list_enc_norm['class'] = class_column
a_list_enc_norm


# In[27]:


#2-2
std_scaler = StandardScaler()
std_feature = std_scaler.fit_transform(a_list_enc.iloc[:,:-1])
a_list_enc_norm_ = pd.DataFrame(std_feature)
a_list_enc_norm_


# In[28]:


a_list_list = a_list_enc_norm.values.tolist()
 
X_data = []
Y_data = []
#3-1
for i in a_list_list:
    X_data.append(i[:-1])
    Y_data.append(i[-1])
#3-2
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, random_state=33)
print("X_train:{0}".format(X_train))
print("\n")
print("X_test:{0}".format(X_test))


# In[133]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import math
#4-1.a
accuracy_ = []
hidden_layer = [1,2,3] #노드 3개

for i in range(3):
    clf = MLPClassifier(random_state = 7777, hidden_layer_sizes = (hidden_layer[i],), solver = 'adam', max_iter = 5000)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    accuracy_.append(accuracy)
    
print("Accuracy for 1 node:{0}".format(round(accuracy_[0],3)));
print("Accuracy for 2 nodes:{0}".format(round(accuracy_[1],3)));
print("Accuracy for 3 nodes:{0}".format(round(accuracy_[2],3)));


# In[138]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#4-1.b
accuracy_02 = []
first_hidden_layer = [4,5,6]
second_hidden_layer = [7,8,9]

for i in range(3):
    for j in range(3):
        clf = MLPClassifier(random_state = 7777, hidden_layer_sizes = (first_hidden_layer[i],second_hidden_layer[j]),max_iter = 5000)
        clf.fit(X_train, Y_train)
        prediction_02 = clf.predict(X_test)
        m_accuracy = accuracy_score(Y_test, prediction_02)
        accuracy_02.append(m_accuracy)
    
for i in range(9):
    print("Accuracy for case {0}:{1}".format(i+1, round(accuracy_02[i],3)))

#4-2
line = ['(4,7)','(4,8)','(4,9)','(5,7)','(5,8)','(5,9)','(6,7)','(6,8)','(6,9)']
plt.bar(line, accuracy_02, color = 'b', width = 0.5)
plt.xlabel('Cases')
plt.ylabel('Accuracy')
plt.show()


# In[30]:


#4-3
line = []

clf = MLPClassifier(activation = 'identity', hidden_layer_sizes = (3,))
clf.fit(X_train, Y_train)

prediction_t = clf.predict(X_test)
res_prediction = accuracy_score(Y_test, prediction_t)
line.append(res_prediction)
print("Accuracy from identity: {}".format(res_prediction))


# In[31]:


clf = MLPClassifier(activation = 'logistic', hidden_layer_sizes = (3,))
clf.fit(X_train, Y_train)

prediction_t = clf.predict(X_test)
res_prediction = accuracy_score(Y_test, prediction_t)
line.append(res_prediction)
print("Accuracy from logistic: {}".format(res_prediction))


# In[32]:


clf = MLPClassifier(activation = 'tanh', hidden_layer_sizes = (3,))
clf.fit(X_train, Y_train)

prediction_t = clf.predict(X_test)
res_prediction = accuracy_score(Y_test, prediction_t)
line.append(res_prediction)
print("Accuracy from tanh: {}".format(res_prediction))


# In[33]:


clf = MLPClassifier(activation = 'relu', hidden_layer_sizes = (3,))
clf.fit(X_train, Y_train)

prediction_t = clf.predict(X_test)
res_prediction = accuracy_score(Y_test, prediction_t)
line.append(res_prediction)
print("Accuracy from relu: {}".format(res_prediction))


# In[34]:


line_2 = ['identity', 'logistic', 'tanh', 'relu']
plt.bar(line_2, line, color = 'b', linewidth = 0.05)
plt.title("Comparison",fontsize = 15)
plt.xlabel("Type of the activation function")
plt.ylabel("Accuracy")
plt.show()


# In[35]:


#4-4
line_3_4 = []
clf = MLPClassifier(hidden_layer_sizes = (3,), momentum = 0)
clf.fit(X_train, Y_train)
prediction_3_4 = clf.predict(X_test)
accuracy_3_4 = accuracy_score(Y_test, prediction_3_4)
line_3_4.append(accuracy_3_4)
print(accuracy_3_4)
print("Confustion Matrix: {}".format(confusion_matrix(Y_test, prediction_3_4)))


# In[36]:


clf = MLPClassifier(hidden_layer_sizes = (3,), momentum = 0.2)
clf.fit(X_train, Y_train)
prediction_3_4 = clf.predict(X_test)
accuracy_3_4 = accuracy_score(Y_test, prediction_3_4)
line_3_4.append(accuracy_3_4)
print(accuracy_3_4)
print("Confustion Matrix: {}".format(confusion_matrix(Y_test, prediction_3_4)))


# In[37]:


clf = MLPClassifier(hidden_layer_sizes = (3,), momentum = 0.4)
clf.fit(X_train, Y_train)
prediction_3_4 = clf.predict(X_test)
accuracy_3_4 = accuracy_score(Y_test, prediction_3_4)
line_3_4.append(accuracy_3_4)
print(accuracy_3_4)
print("Confustion Matrix: {}".format(confusion_matrix(Y_test, prediction_3_4)))


# In[38]:


clf = MLPClassifier(hidden_layer_sizes = (3,), momentum = 0.6)
clf.fit(X_train, Y_train)
prediction_3_4 = clf.predict(X_test)
accuracy_3_4 = accuracy_score(Y_test, prediction_3_4)
line_3_4.append(accuracy_3_4)
print(accuracy_3_4)
print("Confustion Matrix: {}".format(confusion_matrix(Y_test, prediction_3_4)))


# In[39]:


clf = MLPClassifier(hidden_layer_sizes = (3,), momentum = 0.8)
clf.fit(X_train, Y_train)
prediction_3_4 = clf.predict(X_test)
accuracy_3_4 = accuracy_score(Y_test, prediction_3_4)
line_3_4.append(accuracy_3_4)
print(accuracy_3_4)
print("Confustion Matrix: {}".format(confusion_matrix(Y_test, prediction_3_4)))


# In[40]:


line_3_4_mo = ['0' , '0.2', '0.4', '0.6', '0.8']
plt.bar(line_3_4_mo, line_3_4, color = 'b', linewidth = 0.005)
plt.xlabel("Value of momentum")
plt.ylabel("Accuracy")
plt.show()


# In[41]:


#4-5 different learning rates: 1.0, 1.2, 1.4, 1.6
line_3_5 = []
clf = MLPClassifier(hidden_layer_sizes = (3,), learning_rate = 'constant',learning_rate_init = 0.002)
clf.fit(X_train, Y_train)
prediction_3_5 = clf.predict(X_test)
accuracy_3_5 = accuracy_score(Y_test, prediction_3_5)
line_3_5.append(accuracy_3_5)
print(accuracy_3_5)


# In[42]:


clf = MLPClassifier(hidden_layer_sizes = (3,), learning_rate = 'invscaling',learning_rate_init = 0.003)
clf.fit(X_train, Y_train)
prediction_3_5 = clf.predict(X_test)
accuracy_3_5 = accuracy_score(Y_test, prediction_3_5)
line_3_5.append(accuracy_3_5)
print(accuracy_3_5)


# In[43]:


clf = MLPClassifier(hidden_layer_sizes = (3,), learning_rate = 'invscaling',learning_rate_init = 0.0056)
clf.fit(X_train, Y_train)
prediction_3_5 = clf.predict(X_test)
accuracy_3_5 = accuracy_score(Y_test, prediction_3_5)
line_3_5.append(accuracy_3_5)
print(accuracy_3_5)


# In[44]:


clf = MLPClassifier(hidden_layer_sizes = (3,), learning_rate = 'adaptive', learning_rate_init = 0.0016)
clf.fit(X_train, Y_train)
prediction_3_5 = clf.predict(X_test)
accuracy_3_5 = accuracy_score(Y_test, prediction_3_5)
line_3_5.append(accuracy_3_5)
print(accuracy_3_5)


# In[45]:


line_3_5_cases = ['1st', '2nd', '3rd', '4th']
plt.bar(line_3_5_cases, line_3_5, color = 'b', linewidth = 0.05)
plt.xlabel('Cases')
plt.ylabel('Accuracy')
plt.show()


# In[46]:


#5-discretization using a_list_enc
from sklearn.preprocessing import KBinsDiscretizer

disc = KBinsDiscretizer(n_bins=4, encode = 'ordinal',strategy='uniform')
k_scaled = disc.fit_transform(a_list_enc)
a_list_enc_disc = pd.DataFrame(k_scaled)
a_list_enc_disc


# In[56]:


from sklearn.tree import DecisionTreeClassifier
#6-1
a_list_6 = a_list_enc_disc.values.tolist()
 
X_data = []
Y_data = []

for i in a_list_6:
    X_data.append(i[:-1])
    Y_data.append(i[-1])
#3-2
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, random_state=33)

clf_1 = DecisionTreeClassifier(criterion='gini', max_depth=5,random_state=42)
clf_1.fit(X_train, Y_train)
prediction_gini = clf_1.predict(X_test)
accuracy_gini = accuracy_score(Y_test, prediction_gini)

clf_2 = DecisionTreeClassifier(criterion = 'entropy', max_depth =5, random_state = 42)
clf_2.fit(X_train, Y_train)
prediction_entropy = clf_2.predict(X_test)
accuracy_entropy = accuracy_score(Y_test, prediction_entropy)

print("Accuracy by gini:{0}".format(accuracy_gini))
print("Accuracy by entropy:{0}".format(accuracy_entropy))

if accuracy_gini>accuracy_entropy:
    print("Gini is better than entropy.")
elif accuracy_entropy>accuracy_gini:
    print("Entropy is better than gini.")
else:
    print("Same")


# In[73]:


from sklearn.tree import export_graphviz
from sklearn import tree
import graphviz
import pydotplus

##6-2  6-2는 해결하지 못하여 주석처리했습니다.
#dot_data = tree.export_graphviz(clf_1, out_file = None, filled = True, rounded = True, 
                               special_characters = True)
#graph = graphviz.Source(dot_data)
#graph

#6-3
line = [0.001,0.01,0.05,0.4,0.7,1,2,3,6,8]
line_accuracy = []
for i in line:
    clf_3 = DecisionTreeClassifier(criterion = 'gini', max_depth =i, random_state = 42)
    clf_3.fit(X_train, Y_train)
    prediction_6_3= clf_3.predict(X_test)
    accuracy_6_3 = accuracy_score(Y_test, prediction_6_3)
    line_accuracy.append(accuracy_6_3)
    print("Accuracy for 6-3 is {0}, and max_depth is {1}".format(accuracy_6_3,i))


# In[80]:


#6-4
line_n = []
for i in line:
    k = str(i)
    line_n.append(k)

plt.bar(line_n,line_accuracy, color = 'b', width = 0.5)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.show()


# In[88]:


#7-1
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
line_7 = []
line_7.append(KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train).score(X_test, Y_test))
KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train).score(X_test, Y_test)


# In[89]:


line_7.append(KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train).score(X_test, Y_test))
KNeighborsClassifier(n_neighbors=5).fit(X_train, Y_train).score(X_test, Y_test)


# In[91]:


line_7.append(KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train).score(X_test, Y_test))
KNeighborsClassifier(n_neighbors=9).fit(X_train, Y_train).score(X_test, Y_test)


# In[93]:


line_7.append(KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train).score(X_test, Y_test))
KNeighborsClassifier(n_neighbors=13).fit(X_train, Y_train).score(X_test, Y_test)


# In[95]:


line_x = ['1','5','9','13']
plt.bar(line_x, line_7, color = 'b', width = 0.5)
plt.xlabel('n_neighbors')
plt.ylabel('Result')
plt.show()


# In[107]:


#7-2
line_7_2 = []
line_7_2.append(KNeighborsClassifier(weights = 'uniform').fit(X_train, Y_train).score(X_test, Y_test))
KNeighborsClassifier(weights = 'uniform').fit(X_train, Y_train).score(X_test, Y_test)


# In[108]:


line_7_2.append(KNeighborsClassifier(weights = 'distance').fit(X_train, Y_train).score(X_test, Y_test))
KNeighborsClassifier(weights = 'distance').fit(X_train, Y_train).score(X_test, Y_test)


# In[109]:


line_7_2_y = ['uniform', 'distance']
plt.bar(line_7_2_y, line_7_2, color = 'b', width = 0.5)
plt.xlabel('Type of weights')
plt.ylabel('Result')
plt.show()


# In[119]:


#7-3
line_7_3 = []
line_7_3.append(KNeighborsClassifier(p=1).fit(X_train, Y_train).score(X_test, Y_test))
KNeighborsClassifier(p=1).fit(X_train, Y_train).score(X_test, Y_test)


# In[120]:


line_7_3.append(KNeighborsClassifier(p=2).fit(X_train, Y_train).score(X_test, Y_test))
KNeighborsClassifier(p=2).fit(X_train, Y_train).score(X_test, Y_test)


# In[128]:


line_y = ['1','2']
plt.bar(line_y, line_7_3, color = 'b', width = 0.5)
plt.xlabel('Value of p')
plt.ylabel('Result')
plt.show()


# In[ ]:




