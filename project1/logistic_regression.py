import numpy as np
import tqdm

class LogisticRegression():
    """
        A logistic regression model trained with stochastic gradient descent.
    """

    def __init__(self, num_epochs=100, learning_rate=1e-4, batch_size=16, regularization_lambda=0,  verbose=False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.regularization_lambda = regularization_lambda
        self.theta = None
        self.bias = 0
    
    def sigmoid(self,x):
        z = 1/(1 + np.exp(-x))
        return z

    def fit(self, X, Y, val_X, val_Y):
        """
            Train the logistic regression model using stochastic gradient descent.
        """

        num_samples, num_features = X.shape
        num_samples_val, num_features_val = val_X.shape

        self.theta = np.zeros(num_features)

        train_losses, val_losses = [], []

        for epoch in range(self.num_epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                X_batch, Y_batch = X[batch_indices], Y[batch_indices]
                grad_theta, grad_bias = self.gradient(X_batch, Y_batch)

                self.theta -= self.learning_rate * grad_theta
                self.bias -= self.learning_rate * grad_bias

            predictions = self.predict_proba(X)
            loss = -np.mean(Y * np.log(predictions) + (1-Y) * np.log(1-predictions)) + (self.regularization_lambda/2) * np.sum(self.theta**2)

            val_predictions = self.predict_proba(val_X)
            val_loss = -np.mean(val_Y * np.log(val_predictions) + (1-val_Y) * np.log(1-val_predictions)) + (self.regularization_lambda/2) * np.sum(self.theta**2)
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
            
            train_losses.append(loss)
            val_losses.append(val_loss)
        
        return train_losses, val_losses


        #raise NotImplementedError("Not implemented yet")

    def gradient(self, X, Y):
        """
            Compute the gradient of the loss with respect to theta and bias with L2 Regularization.
            Hint: Pay special attention to the numerical stability of your implementation.
        """
        proba = self.predict_proba(X)
        grad_theta = np.mean((proba - Y)@X) + self.regularization_lambda*np.sum(self.theta)
        grad_bias = np.mean(proba - Y)

        return grad_theta, grad_bias
    

    def predict_proba(self, X):
        """
            Predict the probability of lung cancer for each sample in X.
        """
        pred = X.dot(self.theta) + self.bias 
        return self.sigmoid(pred)

    def predict(self, X, threshold=0.5):
        """
            Predict the if patient will develop lung cancer for each sample in X.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
        #raise NotImplementedError("Not implemented yet")