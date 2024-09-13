import numpy as np

class Neuron_Functions: 
    
    @staticmethod
    def ReLU(Z):
        A = np.maximum(Z,0) #element wise max
        return A
    
    @staticmethod
    def softmax(Z):

        Z_shifted = Z - np.max(Z, axis=0, keepdims=True) #to reduce overflow, keep the values near zero
        ex = np.exp(Z_shifted)
        dex = np.sum(ex, axis=0, keepdims=True) 

        A = ex/dex # the formula

        return A

    @staticmethod
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size,Y.max() + 1))
        one_hot_Y[np.arange(Y.size),Y] = 1 #make a matrix like [0,0,1,...], where 1 is the correct answer
        return  one_hot_Y.T #matrix with the results
