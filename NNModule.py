import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from helper import y2indicator, getBinaryfer13Data, sigmoid, sigmoid_cost, error_rate, softmax, init_weight_and_bias


# __author__: Trisha P Malhotra, Ifeoma Nwogu

class NNModule(object):
    def __init__(self):
        pass

    def train(self, X, Y, step_size=10e-7, epochs=10000):
        # Validation data set extracted from the training data
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        K = len(set(Y))

        # Convert outputs of the NN to an indicator matrix
        Ytrain_ind = y2indicator(Y, K)
        Yvalid_ind = y2indicator(Yvalid, K)
        M, D = X.shape
        print(M)
        print(D)

        # HW3_2. Randomly initialize all the hidden weights W's and biases b's
        # Add code here....
        W1, b1 = init_weight_and_bias(D, 1)
        W2, b2 = init_weight_and_bias(D, 1)

        # HW3_3. For the given number of epochs set, implement backpropagation to learn the
        #       weights and append computed costs in the 2 cost arrays
        train_costs = []
        valid_costs = []
        best_validation_error = 1
        for i in range(epochs):
            # HW3_4. Run forward propagation twice; once to calculate P(Ytrain|X)
            #       and Ztrain (activations at hidden layer); second to calculate
            #       P(Yvalid|Xvalid) and Zvalid
            # Add code here....
            """
            check params
            """
            pYtrain, Ztrain = self.forward(X, Y)
            pYvalid, Zvalid = self.forward(Xvalid, Yvalid)

            # HW3_5. Now we do back propagation by first performing
            #       gradient descent using equations (3) and (4) from the HW text
            # Add code here....

            diff = pYtrain - Y
            result = Ztrain.dot(diff)
            self.W2 -= step_size * (result)
            self.b2 -= step_size * (diff)

            # HW3_5b. Now we do back propagation
            # Add code here....
            result2 = (1 - (Ztrain ** 2))
            result3 = W2.dot(result2)
            dJ = diff.dot(result3)
            print(dJ)

            result4 = X.dot(dJ)
            W1 -= step_size.dot(result4)
            print(W1)
            b1 -= step_size.dot(dJ)
            print(b1)

            # HW3_6. Compute the training and validation errors using cross_entropy cost
            #       function; once on the training predictions and once on validation predictions
            #       append errors to appropriate error array 
            # Add code here....
            training_errors = self.cross_entropy(Y, pYvalid)
            validation_errors = self.cross_entropy(Y, pYvalid)

            # target T = Y , and predicted = pY or pY_valid
            # calculating sigmoid_costs_result for validation set
            sigmoid_costs_result_train = sigmoid_cost(Y, pYtrain)
            sigmoid_costs_result_valid = sigmoid_cost(Y, pYvalid)

            # appending in already initialized cost array
            train_costs.append(sigmoid_costs_result_train)
            valid_costs.append(sigmoid_costs_result_valid)

        # HW3_7. Include the best validation error and training and validation classification
        #       rates in your final report
        # Add code here....

        for _ in range(pYvalid):
            validation_error = error_rate(Y, pYvalid)
            if validation_error < validation_error:
                best_validation_error_valid = validation_error

        for _ in range(pYtrain):
            validation_error = error_rate(Y, pYtrain)
            if validation_error < validation_error:
                best_validation_error_train = validation_error

        # HW3_8. Display the training and validation cost graphs in your final report
        # Add code here....

        # graph for training cost
        plt.figure()
        plt.plot(train_costs)
        plt.ylabel('Training Costs graph')
        plt.show()
        # graph for validation cost
        plt.figure()
        plt.plot(valid_costs)
        plt.ylabel('Validation Costs graph')
        plt.show()

        return best_validation_error_train, best_validation_error_valid

    # Implement the forward algorithm
    def forward(self, X, Y):
        # Add code here....
        """
        py: output of the softmax classifier,
        Z : the hidden activations Z, based on which activation function chosen (tanh, sigmoid, or ReLU )
        """
        # softmax classiffier:
        pY = softmax(Y)

        # activation function: sigmoid
        Z = sigmoid(X.dot(self.W) + self.b)
        return pY, Z

    # Implement the prediction algorithm
    def predict(self, X, Y):
        # Add code here....
        prediction_results = self.forward(X, Y)

    # Implement a method to compute accuracy or classification rate
    def classification_rate(self, Y, P):
        # Add code here....
        classification_accuracy = 0
        for _ in range(Y):
            if P != Y:
                classification_accuracy += 1
        return classification_accuracy

    def cross_entropy(self, Y, pY):
        # Add code here....
        return - np.sum(np.multiply(Y, np.log(pY)) + np.multiply((1 - Y), np.log(1 - pY)))


def main():
    # HW3_1. Train a NN classifier on the fer13 training dataset
    # Add code here....

    ## Loading the dataset of datasamples and their class labels
    filename = "fer3and4train.csv"
    X, Y = getBinaryfer13Data(filename)

    ## Calling Train function to learn the weights and bias of the unit.
    foo = NNModule()
    best_validation_err, W, b = foo.train(X, Y)
    print("The Validation Error is", best_validation_err)

    # HW3_9. After your training errors are sufficiently low,
    #       apply the classifier on the test set, 
    #       show the classification accuracy in your final report
    # Add code here....
    test_filename = 'fer3and4test.csv'
    X, Y = getBinaryfer13Data(test_filename)
    test_predicted_Y = foo.predict(X, Y)

    print("Accuracy or Classification Rate ")

    score_result = (foo.score(Y, test_predicted_Y))
    print(score_result)
    # accuracy_rate =  (foo.score_result/ float(Y.size)) * 100.0)


if __name__ == '__main__':
    main()
