#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sklearn

#0
a_list = pd.read_csv('C:\\신현호\\머신러닝\\ionosphere.csv')
a_list


# In[3]:


#1
a_list = a_list.dropna()
le = LabelEncoder()
a_list['class'] = le.fit_transform(a_list['class'])
a_list_enc = a_list
a_list_enc


# In[4]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

min_max_scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = min_max_scaler.fit_transform(a_list_enc.iloc[:,:-1])
a_list_enc_norm = pd.DataFrame(data_scaled)
class_column = a_list_enc['class']
a_list_enc_norm['class'] = class_column
a_list_enc_norm


# In[5]:


#2
a_list_list = a_list_enc_norm.values.tolist()
 
X_data = []
Y_data = []

for i in a_list_list:
    X_data.append(i[:-1])
    Y_data.append(i[-1])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, random_state=33)


# In[6]:


#3
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import math
#3-1
line= [5,7,10,30,100]
result = []
for i in line:
    clf = AdaBoostClassifier(n_estimators = i)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    result.append(accuracy)

for i in range(len(line)):
    print("n_estimator: {}, Accuracy: {}".format(line[i], result[i]))


# In[7]:


#3-2
line_ = ['5', '7', '10', '30', '100']
plt.bar(line_, result, color = 'b', width = 0.5)
plt.xlabel("A value of n_estimators")
plt.ylabel('Accuracy')
plt.show()


# In[8]:


#4
from sklearn.ensemble import RandomForestClassifier
#4-1
line_rf = [2,5,30,50,100]
result_rf = []

for i in line_rf:
    clf_rf = RandomForestClassifier(n_estimators = i)
    clf_rf.fit(X_train, Y_train)
    prediction_rf = clf_rf.predict(X_test)
    accuracy_rf = accuracy_score(Y_test, prediction_rf)
    result_rf.append(accuracy_rf)

for i in range(len(line_rf)):
    print("n_estimator: {}, Accuracy: {}".format(line_rf[i], result_rf[i]))


# In[9]:


#4-2
line_rf_ = ['2', '5', '30', '50', '100']
plt.bar(line_rf_, result_rf, color = 'b', width = 0.5)
plt.xlabel('A value of n_estimators')
plt.ylabel('Accuracy')
plt.show()


# In[13]:


#4-3
#제일 큰 값일 때는 50일 때
clf_50_t = RandomForestClassifier(n_estimators = 50, oob_score = True)
clf_50_t.fit(X_train, Y_train)
prediction_50_t = clf_50_t.predict(X_test)
accuracy_50_t = accuracy_score(Y_test, prediction_50_t)
print("True일 때: {}".format(accuracy_50_t))

clf_50_f = RandomForestClassifier(n_estimators = 50, oob_score = False)
clf_50_f.fit(X_train, Y_train)
prediction_50_f = clf_50_f.predict(X_test)
accuracy_50_f = accuracy_score(Y_test, prediction_50_f)
print("False일 때: {}".format(accuracy_50_f))


# In[14]:


#4-4
line_4_4 = ["auto", "sqrt", "log2"]
result_4_4 = []

for i in line_4_4:
    clf_50_ = RandomForestClassifier(n_estimators = 50, max_features = i,oob_score = True)
    clf_50_.fit(X_train, Y_train)
    prediction_50_ = clf_50_.predict(X_test)
    accuracy_50_ = accuracy_score(Y_test, prediction_50_)
    result_4_4.append(accuracy_50_)
    
for i in range(len(line_4_4)):
    print("max_features: {}, Accuracy: {}".format(line_4_4[i],result_4_4[i]))


# In[15]:


#5-1
from sklearn import svm
clf_svm = svm.SVC()
clf_svm.fit(X_train, Y_train)
prediction_svm = clf_svm.predict(X_test)
accuracy_svm = accuracy_score(Y_test, prediction_svm)
print(accuracy_svm)


# In[16]:


#5-2
line_svm = ['linear', 'poly', 'rbf', 'sigmoid']
result_svm = []

for i in line_svm:
    clf_svm = svm.SVC(kernel = i)
    clf_svm.fit(X_train, Y_train)
    prediction_svm = clf_svm.predict(X_test)
    accuracy_svm = accuracy_score(Y_test, prediction_svm)
    result_svm.append(accuracy_svm)
for i in range(len(line_svm)):
    print("Kernel: {}, Accuracy: {}".format(line_svm[i], result_svm[i]))


# In[18]:


#6-1
from sklearn.cluster import KMeans

line_K = [2,5,7]
result_K = []

for i in line_K:
    kmeans = KMeans(n_clusters = i, algorithm = 'auto', random_state = 0).fit(X_train)
    prediction_k = kmeans.predict(X_test)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    #accuracy_k = accuracy_score(Y_test, prediction_k)
    result_K.append(centroids)
    
for i in range(len(line_K)):
    print("n_clusters: {}, Result: {}".format(line_K[i], result_K[i]))
    


# In[16]:


#6-2
X_test = np.array(X_test)
km = KMeans(n_clusters = 2, algorithm = 'auto', random_state = 0)
prediction_k = km.fit_predict(X_test)
part_cluster_2 = X_test[prediction_k ==0,1]   
result = 0
for j in part_cluster_2:
    result+=j
s = len(part_cluster_2)
ans = result/s
ans


# In[17]:


#6-3
from collections import Counter

cluster_label_1 = X_test[prediction_k == 0,1]
cluster_label_2 = X_test[prediction_k == 1,1]
a_1 = 0
b_1 = 0
a_2 = 0
b_2 = 0
cluster_label__1 = 0
cluster_label__2 = 0
for i in cluster_label_1:
    if i == 0:
        a_1+=1
    elif i == 1:
        b_1+=1
    
for j in cluster_label_2:
    if j==0:
        a_2+=1
    elif j ==1:
        b_2 +=1
if b_1>a_1:
    cluster_label__1 = 1
if b_2>a_2:
    cluster_label__2 = 1

print("First one: {}, Second one: {}".format(cluster_label__1, cluster_label__2))


# In[18]:


#6-4
#두 클러스터의 다수 값이 모두 1이므로, cluster_label을 1로 놓고 풀겠습니다.
cluster_label = 1
labels = km.labels_
s = len(labels)
error = 0 #값이 1이 아닌 경우의 개수

for i in labels:
    if i!=1:
        error+=1
accuracy_6_4 = (s-error)/s

accuracy_6_4


# In[19]:


#6-5
#n_init = 8,10,12 -> 디폴트값은 10
line = [2,3,10]
result__ = []

for i in line:
    kmeans = KMeans(n_clusters = 2, algorithm = 'auto', random_state = 0, n_init = i)
    prediction_k = kmeans.fit_predict(X_test)
    accuracy_k = accuracy_score(Y_test, prediction_k)
    result__.append(accuracy_k)
line_ = ['2','3','10']
plt.bar(line_, result__, color = 'b', width = 0.5)
plt.xlabel('A value of n_init')
plt.ylabel('Accuracy')
plt.show()


# In[24]:


#7-1
from sklearn.mixture import GaussianMixture

line_g = [2,3,7]
result_g = []

for i in line_g:
    gmm = GaussianMixture(n_components = i)
    labels = gmm.fit_predict(X_test)
    accuracy_g = accuracy_score(Y_test, labels)
    result_g.append(accuracy_g)

for j in range(3):
    print("n_components: {}, Accuracy: {}".format(line_g[j], result_g[j]))
line_g_ = ['2','3','7']
plt.bar(line_g_, result_g, color = 'b', width = 0.5)
plt.xlabel('A value of n_components')
plt.ylabel('Accuracy')
plt.show()


# In[27]:


#7-2
gmm_3 = GaussianMixture(n_components = 3).fit(X_test)
labels = gmm_3.predict(X_test)
labels


# In[34]:


#7-3
#7-2에서의 예제를 가지고 문제 풀겠습니다.
probs = gmm_3.predict_proba(X_test)
print(probs[:5,:2])


# In[ ]:




