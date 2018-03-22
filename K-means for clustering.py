
# coding: utf-8

# In[1]:


#K-means for clustering


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
X = np.array([2,2,5,6,5,2.5])
Y = np.array([4,3,2,2,2.5,3.5])
c1 = (2, 4)
c2 = (5, 2)
clus1 = []
clus2 = []
def distance(c1, c2, x, y):
    for i in range(len(x)):
        m = ((c1[0] - x[i])**2 + ( c1[1] - y[i])**2)**1/2
        n = ((c2[0] - x[i])**2 + ( c2[1] - y[i])**2)**1/2
        if m < n:
            clus1.append([x[i], y[i]])
            c1 = (np.mean([k[0] for k in clus1]),
                  np.mean([k[1] for k in clus1]))
            print(c1)
        else:
            clus2.append([x[i], y[i]])
            c2 = (np.mean([k[0] for k in clus2]) , 
                  np.mean([k[1] for k in clus2]))
            print(c2)
    return c1, c2


# In[16]:


c1_final, c2_final = distance(c1, c2, X, Y)


# In[15]:


plt.scatter(X,Y)
plt.scatter(c1_final[0],c1_final[1])
plt.scatter(c2_final[0], c2_final[1])
plt.show()

