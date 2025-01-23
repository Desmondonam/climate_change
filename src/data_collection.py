import pandas as pd
import numpy as np
from typing import Dict, Any

class DataCollector:
    def __init__(self, temperature_source: str = None, co2_source: str = None):
        """
        Initialize data collector with optional custom data sources
        
        Args:
            temperature_source (str): Path to temperature data
            co2_source (str): Path to CO2 emissions data
        """
        self.temperature_data = None
        self.co2_data = None
        
        # Default data sources (replace with actual URLs/paths)
        self.default_temp_source = temperature_source or "https://github.com/Desmondonam/climate_change/blob/main/data/observed-annual-average.csv"
        self.default_co2_source = co2_source or "https://example.com/co2_data.csv"
    
    def collect_temperature_data(self) -> pd.DataFrame:
        """
        Collect global temperature data
        
        Returns:
            pd.DataFrame: Cleaned temperature dataset
        """
        try:
            self.temperature_data = pd.read_csv(self.default_temp_source)
            
            # Basic data cleaning
            self.temperature_data['Year'] = pd.to_datetime(self.temperature_data['Year'], format='%Y')
            self.temperature_data.dropna(inplace=True)
            
            return self.temperature_data
        except Exception as e:
            print(f"Error collecting temperature data: {e}")
            return pd.DataFrame()
    
    def collect_co2_emissions(self) -> pd.DataFrame:
        """
        Collect CO2 emissions data
        
        Returns:
            pd.DataFrame: Cleaned CO2 emissions dataset
        """
        try:
            self.co2_data = pd.read_csv(self.default_co2_source)
            
            # Basic data cleaning
            self.co2_data['Year'] = pd.to_datetime(self.co2_data['Year'], format='%Y')
            self.co2_data.dropna(inplace=True)
            
            return self.co2_data
        except Exception as e:
            print(f"Error collecting CO2 emissions data: {e}")
            return pd.DataFrame()
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge temperature and CO2 emissions datasets
        
        Returns:
            pd.DataFrame: Combined dataset
        """
        if self.temperature_data is None:
            self.collect_temperature_data()
        
        if self.co2_data is None:
            self.collect_co2_emissions()
        
        merged_data = pd.merge(
            self.temperature_data, 
            self.co2_data, 
            on='Year', 
            how='inner'
        )
        
        return merged_data