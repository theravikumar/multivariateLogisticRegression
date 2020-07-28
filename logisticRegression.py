#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class LogisticRegression:
    def __init__ (self,lr=0.0001, itera=500000):
        self.lr = lr
        self.itera = itera
        self.w = None
        self.b = None
    def fit(self,X,Y):
        n_data , n_features =X.shape
        self.w = np.zeros(n_features)   
        self.b = 0
        for i in range (self.itera):
            Y_pred = np.dot(X,self.w) + self.b 
            Y_pred = self.sigmoid(Y_pred)
            loss = (1/n_data) * (Y_pred - Y)            
            dw = (1/n_data) * np.dot(X.T,loss)
            db = (1/n_data) * np.sum(loss)            
            self.w -= (self.lr * dw)
            self.b -= (self.lr * db)
    def sigmoid(self,x):
        return (1/(1+np.exp(-x)))
        
    def predict(self,X):
        y_pred = np.dot(X,self.w)+self.b
        y_pred = self.sigmoid(y_pred)
        y_class_pred = [1 if i>0.5 else 0 for i in y_pred]
        return y_class_pred
    
    def accuracy(self,Y_act,Y_predi):
        counter = 0
        for i in range (len(Y_act)):
            if Y_act[i]==Y_predi[i]:
                counter += 1
        return ((counter/len(Y_act))*100)


df = pd.read_csv("/home/jarvis/python/sportscar_choice_long.csv")



Convert = {'yes': 1,'no': 0}
Transe = {'manual':1,'auto':2}
Segment = {'basic':1,'racer':2,'fun':3}




df.convert = [Convert[item] for item in df.convert]
df.trans = [Transe[item] for item in df.trans]
df.segment = [Segment[item] for item in df.segment]





data = df.iloc[:,:].values





np.random.shuffle(data)

n_trainData = data.shape[0] * 0.8
X_train = data[:int(n_trainData),:-1]


X_test = data[int(n_trainData):,:-1]



Y_train = data[:int(n_trainData),-1]


Y_test = data[int(n_trainData):,-1]



regression = LogisticRegression()
regression.fit(X_train,Y_train)
Y_predicted = regression.predict(X_test)
# print(Y_predicted)
Accu = regression.accuracy(Y_test,Y_predicted) 
print(Accu)




