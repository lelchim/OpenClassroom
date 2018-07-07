@@ -0,0 +1,80 @@

# coding: utf-8

# In[46]:

import pandas as pd
import numpy as np
from sklearn import neighbors


# In[48]:

data = pd.read_csv('C:\\Users\\chrys\\Documents\\winequality-red.csv', sep=";")


# In[49]:

X = data.as_matrix([data.columns[:-1]])#prendre tout les colonne excepté la dernière

y = data.as_matrix([data.columns[-1]])#prendre la dernière colonne  
y.flatten()#transformer en une seule colonne


# In[50]:

from sklearn import model_selection

X_train, X_test, y_train, y_test  = model_selection.train_test_split(X,y, test_size=0.3)


# In[30]:




# In[55]:

from sklearn import neighbors, metrics
from matplotlib import pyplot as plt

knn= neighbors.KNeighborsRegressor(n_neighbors = 12)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
    


# In[57]:


sizes = {} # clé : coordonnées ; valeur : nombre de points à ces coordonnées
for (yt, yp) in zip(list(y_test), list(y_pred)):
    if (yt, yp) in sizes.keys():
        sizes[(yt, yp)] += 1
    else:
        sizes[(yt, yp)] = 1

keys = sizes.keys()
plt.scatter([k[0] for k in keys], # vraie valeur (abscisse)
[k[1] for k in keys], # valeur predite (ordonnee)
s=[sizes[k] for k in keys], # taille du marqueur
color='coral')


# In[36]:

param = range(2,15)
RMSE = []
for i in param:
    knn= neighbors.KNeighborsRegressor(n_neighbors = i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    RMSE.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
plt.plot(param, RMSE)
plt.show()


# In[ ]:


