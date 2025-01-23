from src.data_collection import DataCollector
from src.data_preprocessing import DataPreprocessor
from src.model_training import ClimateChangeModel
from src.dashboard import ClimateDashboard

def main():
    # Data Collection
    collector = DataCollector()
    climate_data = collector.merge_datasets()
    
    # Data Preprocessing
    preprocessor = DataPreprocessor(climate_data)
    preprocessor.prepare_ml_dataset(target_column='Annual Mean')
    
    # Model Training
    model = ClimateChangeModel(
        preprocessor.X_train, 
        preprocessor.X_test, 
        preprocessor.y_train, 
        preprocessor.y_test
    )
    model.train_model()
    
    # Model Evaluation
    performance = model.evaluate_model()
    print("Model Performance:", performance)
    model.save_model()
    
    # Dashboard
    dashboard = ClimateDashboard(climate_data)
    dashboard.run_server()

if __name__ == "__main__":
    main()