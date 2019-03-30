import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from helper import getBinaryfer13Data, sigmoid, sigmoid_cost, error_rate, init_weight_and_bias, cost

# __author__: Trisha P Malhotra, Ifeoma Nwogu

class LRModule(object):
    # global variable
    """
    CONFIRM
    """

    global W, b

    def __init__(self):
        pass

    def train(self, X, Y, step_size=10e-7, epochs=10000):
        # Validation data set extracted from the training data
        X, Y = shuffle(X, Y)

        # validation dataset :
        Xvalid, Yvalid = X[-1000:], Y[-1000:]

        X, Y = X[:-1000], Y[:-1000]
        N, D = X.shape

        # HW3_2. Initialize the weights W to small random numbers (variance - zero);
        # also initialize the bias b to zero
        # Add code here...

        # uncomment if need be
        #self.W, self.b = init_weight_and_bias(D, len(Y))
        #"""
        X = np.insert(X, 0, 1, axis=1)
        self.W = np.ones(X.shape[1])
        self.b = X.shape[0]

        #"""

        # HW3_3. For the given number of epochs set, learn the weights
        costs = []
        validation_error = 1
        for i in range(epochs):
            # HW3_4. Do forward propagation to calculate P(Y|X)
            # Add code here....

            # predicted Y for given X
            pY = self.forward(X, Y)

            # HW3_5. Perform gradient descent using equations (1) and (2) from the HW text
            # Add code here....
            diff = pY - Y

            self.W -= step_size * (X.dot(diff))
            self.b -= step_size * (diff)

            # HW3_6. Using the validation data, compute P(Y|X_valid) using the forward algorithm
            #       Compute the sigmoid costs and append to array costs
            #       Check to set best_validation_error
            # Add code here....
            pY_valid = self.forward(Xvalid, Yvalid)

            # target T = Y , and predicted = pY or pY_valid
            # calculating sigmoid_costs_result for validation set
            sigmoid_costs_result = sigmoid_cost(Y, pY_valid)

            # appending in already initialized cost array
            costs.append(sigmoid_costs_result)

            # HW3_7. Include the value for the best validation error in your final report
            # Add code here....

            for _ in range(pY_valid):
                validation_error = error_rate(Y, pY_valid)
                if validation_error < validation_error:
                    best_validation_error = validation_error

        # HW3_8. Display the graph of the validation cost in your final report
        # Add code here....
        plt.figure()
        plt.plot(costs)
        plt.ylabel('Costs graph')
        plt.show()

        """
        CONFIRM
        """
        return best_validation_error

    # . Implement the forward algorithm
    def forward(self, X, Y):
        # Add code here....
        return sigmoid(X.dot(self.W) + self.b)

    # . Implement the prediction algorithm, calling forward
    def predict(self, X, Y):
        # Add code here....
        prediction_results = self.forward(X, Y)

    # . Implement a method to compute accuracy or classification rate
    def score(self, Y, test_predicted_Y):
        # Add code here....

        score_count = 0
        for _ in range(Y):
            if test_predicted_Y != Y:
                score_count += 1
        return score_count


def main():
    # HW3_1. Train a LR classifier on the fer13 training dataset
    # Add code here....

    ## ## Loading the dataset of datasamples and their class labels
    filename = "fer3and4train.csv"
    X, Y = getBinaryfer13Data(filename)

    ## Calling Train function to learn the weights and bias of the unit.
    foo = LRModule()
    best_validation_err = foo.train(X, Y)
    print("The Validation Error is", best_validation_err)

    # HW3_9. After your training errors are sufficiently low,
    #       apply the classifier on the test set,
    #       show the classification accuracy in your final report
    # Add code here....

    test_filename = 'fer3and4test.csv'
    X, Y = getBinaryfer13Data(test_filename)
    test_predicted_Y = foo.predict(X, Y)
    print(X)
    print(Y)
    print("Accuracy rate ")

    score_result= (foo.score(Y, test_predicted_Y))
    print(score_result)
    #accuracy_rate =  (foo.score_result/ float(Y.size)) * 100.0)



if __name__ == '__main__':
    main()
