
# coding: utf-8

# # Importing libraries 

#    

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the dataset

#       

# In[11]:


url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#column names
names =['sepal-length','sepal-width', 'petal-length', 'petal-width','Class']
#reading dataset to pandas dataframe
dataset=pd.read_csv(url, names=names)


# # Preprocessing

#     
#     

# In[ ]:


#x==>first 4 cols
#y==>labels
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values


# # Train Test Split

#                  

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


# # Feature Scaling

#               

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


# # Training & Predictions

#                

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5)
#P.S. the only parameter it takes is K
classifier.fit(x_train,y_train)
y_pred =classifier.predict(x_test)


# # Evaluating the Algorithm

#               
#               

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# # Comparing error rate with the K value

# In[ ]:


error =[]
#Calculating the error
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    prediction=knn.predict(x_test)
    error.append(np.mean(prediction!=y_test))


# In[ ]:


#plotting the error
plt.figure(figsize=(10,5))
plt.plot(range(1,40),error, color='red', marker ='x', markerfacecolor='black',markersize=7)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

