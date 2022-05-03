#!/usr/bin/env python
# coding: utf-8

# ## Usha Padma Rachakonda
# #### Implementation of the Gradient descent and Stochastic gradient descent algoritm with logistic regrssion as function
# ### Combinatorial Decision Making and Optimization course's second module taught by Vittorio Maniezzo
# 

#  Predict whether a patient should be diagnosed with Heart Disease or not. This is a binary outcome, 1 indicates  patient diagnosed with Heart Disease and -1 indicates patient not diagnosed with Heart Disease. Here experiment with various optimizations algorithms and see which yields greatest accuracy.

# In[1]:


# Loading Packages
import numpy as np #for mathematical calculation
import pandas as pd
import random
from math import exp
from sklearn.metrics import mean_squared_error


# In[2]:


#loading the Dataset to pandas DataFrame
df=pd.read_csv("D:/Subjects/First year/CDMO/heart_disease.csv")
df


# In[3]:


# Checking the missing values
df.isnull().sum()


# In[4]:


#Returns the first 5 rows data from the entire dataset
df['heartdisease::category|-1|1'].replace([1,-1],[1,0],inplace=True)
df.head(5)


# In[5]:


x=df.iloc[:,0:13].values
x


# In[6]:


y=df.iloc[:,-1].values
y


# In[7]:


from sklearn import model_selection
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[8]:


print(x_train.shape)


#  ## Gradient Descent algorithm with Logistic Regression Function

# ### Gradient Descent:
# Gradient descent is an optimization algorithm that's used when training a machine learning model. It's based on a convex function and tweaks its parameters iteratively to minimize a given function to its local minimum. You start by defining the initial parameter's values and from there gradient descent uses calculus to iteratively adjust the values so they minimize the given cost-function. To understand this concept full, it's important to know about gradients.

# In[21]:


#gradient decient algoritm with logistic regrssion as function:
loss_gd=[]
class gd() :
    def __init__( self, learning_rate, iterations ) :        
        self.learning_rate = learning_rate        
        self.iterations = iterations
             
    def fit( self, X, Y ) :             
        self.m,self.n = X.shape   
              
        self.W = np.zeros( self.n )        
        self.b = 0        
        self.X = X        
        self.Y = Y
                  
        for i in range( self.iterations ) :            
            self.update_weights()            
        return self
      
    def update_weights( self ) :           
        A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
                
        tmp = ( A - self.Y.T )
        dW = np.dot( self.X.T, tmp ) / self.m         
        db = np.sum( tmp ) / self.m 
          
        self.W = self.W - self.learning_rate * dW    
        self.b = self.b - self.learning_rate * db
        loss = mean_squared_error(self.Y, (1/( 1 + np.exp(-(self.X.dot( self.W ) + self.b )))))
        loss_gd.append(np.mean(loss))
          
        return self
      
      
    def predict( self, X ) :    
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )        
        Y = np.where( Z >= 0.5, 1, 0 )        
        return Y


# In[25]:


import warnings
warnings.filterwarnings( "ignore" )
model = gd( learning_rate = 0.0001, iterations = 13000 )     
model.fit( x_train, y_train )


# In[26]:


y_pred = model.predict( x_test )


# In[27]:


# The accuracy
accuracy = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        accuracy += 1
print(f"Accuracy = {accuracy / len(y_pred)}")


# ## Stochastic Gradient Descent algorithm with Logistic Regression Function

# ### Stochastic Gradient Descent:
# By contrast, stochastic gradient descent (SGD) does this for each training example within the dataset, meaning it updates the parameters for each training example one by one. Depending on the problem, this can make SGD faster than batch gradient descent. One advantage is the frequent updates allow us to have a pretty detailed rate of improvement.
# 
# The frequent updates, however, are more computationally expensive than the batch gradient descent approach. Additionally, the frequency of those updates can result in noisy gradients, which may cause the error rate to jump around instead of slowly decreasing.

# In[17]:


#sgd using logisticc regression function as function
loss_sgd=[]
class sgd() :
    def __init__( self, learning_rate, iterations ) :        
        self.learning_rate = learning_rate        
        self.iterations = iterations
             
    def fit( self, X, Y,size ) :             
        self.m,self.n = X.shape   
              
        self.W = np.zeros( self.n )        
        self.b = 0        
        self.X = X        
        self.Y = Y
        self.size=size
        self.update_weights()
                  
    def batch(self, X, Y, size):
        n1=X.shape[0]
        x=list(range(n1))
        random.shuffle(x)
        for size_i,i in enumerate(range(0,n1,size)):
            j=np.array(x[i:min(i+size,n1)])
            yield size_i,X[j,:],Y[j]
      
    def update_weights( self ) :           
        for i in range(self.iterations):       
            for i,X_batch,Y_batch in self.batch(self.X,self.Y,self.size):
                #s1=len(X_batch)
                tmp=(1/( 1 + np.exp(-(X_batch.dot( self.W ) + self.b )))-(Y_batch.T))
                dW = np.dot(X_batch.T, tmp ) / self.m         
                db = np.sum( tmp ) / self.m 
          
                self.W = self.W - self.learning_rate * dW    
                self.b = self.b - self.learning_rate * db
            loss = mean_squared_error(Y_batch, (1/( 1 + np.exp(-(X_batch.dot( self.W ) + self.b )))))
            loss_sgd.append(loss)
          
        return self
        
      
    def predict( self, X ) : 
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )        
        Y = np.where( Z >= 0.5, 1, 0 )        
        return Y


# In[18]:


import warnings
warnings.filterwarnings( "ignore" )
model1= sgd( learning_rate = 0.01, iterations = 13000)      
model1.fit( x_train, y_train,5 )


# In[19]:


y_pred = model1.predict( x_test )


# In[20]:


# The accuracy
accuracy = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        accuracy += 1
print(f"Accuracy = {accuracy / len(y_pred)}")


# In[ ]:




