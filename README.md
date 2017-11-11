# Deep Neural Network Class Implementation: Step by Step

This is a work derived from the first course of the coursera [deeplearning.ai](deeplearning.ai) specialization.
I caught all the functions that was implemented during the course, and use these functions to scructure the code using Object Oritend aproach to 
create a simple **Deep Neural Network Class** in python, the code was written/organized 
with the focus on learning and clarity, for whoever who wants to know how to implement a Neural Net Class from scratch

- In this project, there is a class with all the methods required to train a deep neural network:
    - NerutalNet.py
- There is a lib with all the functions to make the computations of foward propagation, back propagation and Gradient Descent:
    - dnnutils.py    
- Then NerutalNet class will be used to build a deep neural network for Cat no Cat image classification as an example

**How to Use**
- To train a neural net:
```python
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet

train_x = None #Numpy matrix (n,m)
train_y = None #Numpy matrix (1,m)

test_x = None #Numpy matrix (n,mt)
test_y = None #Numpy matrix (1,mt)

layers_dims = [50, 20, 7, 5, 1] # 5-layer model

# Create a NeuralNet object with hiperparameters, X and Y values
neuralNet = NeuralNet(train_x, train_y, layers_dims, learning_rate = 0.007, num_iterations = 2500, print_cost = True)
# Fit the model
neuralNet.fit()
# Plot the cost over time
neuralNet.plotCost()
```
- To predict values:
```python
pred_train = neuralNet.predict(train_x) #To get the predict values
neuralNet.binaryScore(pred_train, train_y) #To print Accuracy, Precision, Recall, Confusion matrix
```
    
**Based on**

All the code base, notations, functions was based on the material from the [Deep Learning Specialization on Coursera](https://www.coursera.org/specializations/deep-learning).
 