import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)
            
            # Gradient descent
            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return [1 if pred >= 0.5 else 0 for pred in predictions]

# Example usage:
if __name__ == "__main__":
    # Sample dataset
    X = np.array([[2, 3], [4, 5], [6, 7], [8, 9]])
    y = np.array([0, 0, 1, 1])
    
    # Initialize and fit logistic regression model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Predictions
    test_data = np.array([[1, 2], [5, 6]])
    predictions = model.predict(test_data)
    print("Predictions:", predictions)
