import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize data preprocessor
        
        Args:
            data (pd.DataFrame): Input dataset
        """
        self.original_data = data
        self.preprocessed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def feature_engineering(self) -> pd.DataFrame:
        """
        Perform feature engineering
        
        Returns:
            pd.DataFrame: Engineered features
        """
        df = self.original_data.copy()
        
        # Create rolling averages
        df['temp_5yr_avg'] = df['Annual Mean'].rolling(window=5).mean()
        df['co2_5yr_avg'] = df['Annual CO₂ emissions'].rolling(window=5).mean()
        
        # Lag features
        df['temp_last_year'] = df['Annual Mean'].shift(1)
        df['co2_last_year'] = df['Annual CO₂ emissions'].shift(1)
        
        return df.dropna()
    
    def scale_features(self, features: list) -> np.ndarray:
        """
        Scale numerical features
        
        Args:
            features (list): List of feature column names
        
        Returns:
            np.ndarray: Scaled feature matrix
        """
        scaler = StandardScaler()
        return scaler.fit_transform(self.preprocessed_data[features])
    
    def prepare_ml_dataset(self, target_column: str, test_size: float = 0.2):
        """
        Prepare dataset for machine learning
        
        Args:
            target_column (str): Column to predict
            test_size (float): Proportion of test dataset
        """
        self.preprocessed_data = self.feature_engineering()
        
        # Select features and target
        features = [
            'Annual Mean', 'Annual CO₂ emissions', 
            'temp_5yr_avg', 'co2_5yr_avg', 
            'temp_last_year', 'co2_last_year'
        ]
        
        X = self.scale_features(features)
        y = self.preprocessed_data[target_column]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )