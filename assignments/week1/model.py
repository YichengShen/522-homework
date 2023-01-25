import numpy as np
import random


class LinearRegression:

    """
    A linear regression model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = np.empty(1)
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculates the closed-form solution of the linear regression.

        Arguments:
            X (numpy.ndarray): feature matrix with dimensions NxD, D: features, N: examples.
            y (numpy.ndarray): labels.

        Returns:
            None
        """

        num_examples = X.shape[0]

        # Add bias term to X
        X = np.hstack((np.ones((num_examples, 1)), X))

        y = y.reshape((-1, 1))

        # Check invertibility
        if np.linalg.det(X.T @ X) != 0:
            params = np.linalg.inv(X.T @ X) @ X.T @ y
            self.w = params[1:]
            self.b = params[0]
        else:
            print("LinAlgError: Matrix is Singular. No analytical solution.")
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates predicted labels.

        Arguments:
            X (numpy.ndarray): feature matrix with dimensions NxD, D: features, N: examples.

        Returns:
            numpy.ndarray: predicted labels.
        """
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Performs gradient descent to fit the model.

        Arguments:
            X (numpy.ndarray): feature matrix with dimensions NxD, D: features, N: examples.
            y (numpy.ndarray): labels.
            lr (float): learning rate.
            epochs (int): number of epochs to train.

        Returns:
            None
        """

        num_examples = X.shape[0]

        # Initialize weights
        self.w = np.random.randn(X.shape[1])
        self.b = random.random()

        for _ in range(epochs):
            y_pred = X @ self.w + self.b
            grad_w = (-2 / num_examples) * (X.T @ (y - y_pred))
            grad_b = (-2 / num_examples) * np.sum(y - y_pred)
            self.w -= lr * grad_w
            self.b -= lr * grad_b
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b
