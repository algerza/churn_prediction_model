


class FibonacciSequence:
    def __init__(self, n):
        self.n = n
        self.sequence = self._generate_sequence()
        
    def _generate_sequence(self):
        # Initialize the first two numbers in the sequence
        fib_sequence = [0, 1]
        
        # Generate the rest of the sequence up to the specified length n
        for i in range(2, self.n):
            # Calculate the next number in the sequence as the sum of the previous two numbers
            next_number = fib_sequence[-1] + fib_sequence[-2]
            fib_sequence.append(next_number)
            
        return fib_sequence
    
    def get_sequence(self):
        return self.sequence



# Required imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Class definition
class ChurnModel:
    
    def __init__(self, data):
        self.data = data
        
    def preprocess_data(self):
        """
        Preprocesses the data by performing scaling and splitting it into training and testing sets.
        """
        # Perform scaling
        scaler = StandardScaler()
        X = self.data.drop(['churn'], axis=1)
        y = self.data['churn']
        X_scaled = scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    def train_model(self):
        """
        Trains a logistic regression model using the training data.
        """
        # Train model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate_model(self):
        """
        Evaluates the performance of the model using the testing data.
        """
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:,1]
        
        # Calculate metrics
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.precision = precision_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred)
        self.f1 = f1_score(self.y_test, y_pred)
        self.auc = roc_auc_score(self.y_test, y_prob)
        
    def predict_churn(self, data):
        """
        Predicts the churn probability for new data.
        """
        # Preprocess data
        X_scaled = scaler.transform(data)
        
        # Make prediction
        churn_prob = self.model.predict_proba(X_scaled)[:,1]
        
        return churn_prob

# Description
"""
The ChurnModel class is designed to preprocess data
"""