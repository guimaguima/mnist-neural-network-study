from base64 import b16decode
import numpy as np
import pandas as pd
from neuron_functions import Neuron_Functions 
from matplotlib import pyplot as plt


class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.rand(10,784) * 0.05 #random min number
        self.W2 = np.random.rand(10,10) * 0.05
        self.b1 = np.random.rand(10,1) * 0.05
        self.b2 = np.random.rand(10,1) * 0.05


    def forward_prop(self,X):
        self.Z1 = self.W1.dot(X) + self.b1 #first layer
        self.A1 = Neuron_Functions.ReLU(self.Z1)
        
        self.Z2 = self.W2.dot(self.A1) + self.b2 #second layer
        self.A2 = Neuron_Functions.softmax(self.Z2)

        return self.A2 #output
        
    def update_params(self,dW1,db1,dW2,db2,eta):
        self.W1 = self.W1 - eta * dW1 #update values
        self.b1 = self.b1 - eta * db1
        self.W2 = self.W2 - eta * dW2
        self.b2 = self.b2 - eta * db2

        return self.W1,self.b1,self.W2,self.b2

    def backward_prop(self,X,Y,eta):
        one_hot_Y = Neuron_Functions.one_hot(Y) #matrix with the answer
        
        dZ2 = 2*(self.A2 - one_hot_Y)/Y.size #derivate of softmax formula reduced
        #also, divide by Y.size, the m amountof training
        dW2 = dZ2.dot(self.A1.T)
        db2 = np.sum(dZ2, axis = 1, keepdims = True)# sum all the columns and keep the dimensions
        dA1 = self.W2.T.dot(dZ2)
        
        dZ1 = dA1 * (self.Z1 > 0).astype(int) #ReLU derivate
        dW1 = dZ1.dot(X.T)
        db1 = np.sum(dZ1, axis = 1, keepdims = True)

        return self.update_params(dW1,db1,dW2,db2,eta)

    
    def get_predictions(self,A2):
        return np.argmax(A2, 0) #make a vector with the predicted number  

    def get_accuracy(self,predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self,X, Y, eta, iterations):
        for i in range(iterations):
            A2 = self.forward_prop(X) #output
            W1, b1, W2, b2 = self.backward_prop(X, Y,eta) #new variables
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(A2)
                print(self.get_accuracy(predictions,Y))#accuracy of the current output
        return W1, b1, W2, b2
    
    def make_predictions(self,X,Y_test=False):
        A2 = self.forward_prop(X)#output
        predictions = self.get_predictions(A2)
        if Y_test is not False:
            accuracy = self.get_accuracy(predictions,Y_test)
            print(f'Accuracy: {accuracy}')#the accuracy of the generalization process
        return predictions