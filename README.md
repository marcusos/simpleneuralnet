# Deep Neural Network Class Implementation: Step by Step

This is a work derivated from the first course of the coursera [deeplearning.ai](deeplearning.ai) specialization.
I caught all the functions that are implemented during the course, and use these functions to scructure the code using Object Oritend aproach to 
create a simple **Deep Neural Network Class** in python, the code was the code was written/organized 
with the focus on learning and clarity, for whoever who wants to know how to implement a Neural Net Class from scratch

- In this project, there is a class with all the methods required to train a deep neural network:
    - NerutalNet.py
- There is a lib with all the functions to make the computations of foward propagation, back propagation and Gradient Descent:
    - dnnutils.py    
- Then NerutalNet class will be used to build a deep neural network for Cat no Cat image classification as an example

**How to Use**

**Notation of the project**:
- Superscript $[l]$ denotes a quantity associated with the $l^{th}$ layer.
    - Example: $a^{[L]}$ is the $L^{th}$ layer activation. $W^{[L]}$ and $b^{[L]}$ are the $L^{th}$ layer parameters.
- Superscript $(i)$ denotes a quantity associated with the $i^{th}$ example.
    - Example: $x^{(i)}$ is the $i^{th}$ training example.
- Lowerscript $i$ denotes the $i^{th}$ entry of a vector.
    - Example: $a^{[l]}_i$ denotes the $i^{th}$ entry of the $l^{th}$ layer's activations).
    
**Based on**
 All the code base, notations, functions was based on the material from the [Deep Learning Specialization on Coursera](https://www.coursera.org/specializations/deep-learning).
 