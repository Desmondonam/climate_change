import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class ClimateChangeModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize climate change prediction model
        
        Args:
            X_train (np.ndarray): Training feature matrix
            X_test (np.ndarray): Testing feature matrix
            y_train (pd.Series): Training target variable
            y_test (pd.Series): Testing target variable
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.predictions = None
    
    def train_model(self, n_estimators: int = 100, random_state: int = 42):
        """
        Train Random Forest Regression model
        
        Args:
            n_estimators (int): Number of trees
            random_state (int): Seed for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state
        )
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self) -> dict:
        """
        Evaluate model performance
        
        Returns:
            dict: Model performance metrics
        """
        self.predictions = self.model.predict(self.X_test)
        
        return {
            'mse': mean_squared_error(self.y_test, self.predictions),
            'r2_score': r2_score(self.y_test, self.predictions)
        }
    
    def save_model(self, filepath: str = 'models/climate_model.pkl'):
        """
        Save trained model
        
        Args:
            filepath (str): Path to save model
        """
        joblib.dump(self.model, filepath)
    
    def predict_future(self, future_data: np.ndarray):
        """
        Make future predictions
        
        Args:
            future_data (np.ndarray): Future feature matrix
        
        Returns:
            np.ndarray: Future predictions
        """
        return self.model.predict(future_data)