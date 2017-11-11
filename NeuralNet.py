import numpy as np
import dnnutils as dnnutils
import matplotlib.pyplot as plt
#from testCases_v3 import *
#from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward


class NeuralNet(object):

    def __init__(self, X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
        self.X = X
        self.Y = Y
        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost

        self.costs = None
        self.parameters = None

    def binaryScore(self, P, Y):
        m = Y.shape[1]
        positivesIdx = Y == 1
        negativesIdx = Y == 0

        temp = P[positivesIdx]
        tp = temp[temp==1].size
        fn = temp[temp == 0].size

        temp = P[negativesIdx]
        tn = temp[temp == 0].size
        fp = temp[temp == 1].size

        confusion_matrix = [[tp,fp],[fn,tn]]

        accuracy = (tp + tn)/m
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)

        # print results
        # print ("predictions: " + str(p))
        # print ("true labels: " + str(y))
        print("Accuracy: " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("Confusion matrix: " + str(confusion_matrix))
        print("tp: ", tp, "fp: ", fp, "fn :", fn, "tn: ", tn)

    def predict(self, X):
        """
        This function is used to predict the results of a  L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        n = len(self.parameters) // 2  # number of layers in the neural network
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches = dnnutils.L_model_forward(X, self.parameters)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        return p

    def plotCost(self):
        # plot the cost
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()

    def fit(self):  # lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        np.random.seed(1)
        self.costs = []  # keep track of cost

        # Parameters initialization.
        self.parameters = dnnutils.initialize_parameters_deep(self.layers_dims)

        # Loop (gradient descent)
        for i in range(0, self.num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = dnnutils.L_model_forward(self.X, self.parameters)
            # Compute cost.
            cost = dnnutils.compute_cost(AL, self.Y)
            # Backward propagation.
            grads = dnnutils.L_model_backward(AL, self.Y, caches)
            # Update parameters.
            self.parameters = dnnutils.update_parameters(self.parameters, grads, self.learning_rate)

            # Print the cost every 100 training example
            if self.print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))
            if self.print_cost and i % 100 == 0:
                self.costs.append(cost)


        return self.parameters