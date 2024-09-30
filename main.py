# main.py
# Author: Valentin Mehedinteanu

import signal
import sys

from src.data_loader import DataLoader, DataLoaderError
from src.feature_engineering import FeatureEngineering, FeatureEngineeringError
from src.model_training import ModelTraining, ModelTrainingError
from src.logger import Logger



def signal_handler(_sig, _frame):
    """
    Handle SIGINT (Ctrl-C) signals to exit the program gracefully.

    Args:
        _sig: Signal number (unused).
        _frame: Current stack frame (unused).

    Returns:
        None
    """
    print("\nSIGINT received. Exiting gracefully...")
    sys.exit()


def main():
    """
    Main function to orchestrate data loading, cleaning, feature engineering, and model training.
    """
    signal.signal(signal.SIGINT, signal_handler)

    logger = Logger.get_logger_for_module(__name__, 'logs/main.log')
    logger.info("=== Starting main pipeline ===")

    try:
        # Step 1: Data Loading and Cleaning
        logger.info("Step 1: Data Loading and Cleaning")
        data_loader = DataLoader(file_path='data/us_stock_data.csv')
        data_loader.load_raw_data()
        data_loader.clean_data()
        data_loader.save_cleaned_data()

        cleaned_df_no_date = data_loader.load_cleaned_data()

        # Step 2: Feature Engineering
        logger.info("Step 2: Feature Engineering")
        feature_engineering = FeatureEngineering(df=cleaned_df_no_date)

        stock_names = [
            'Natural_Gas', 'Crude_oil', 'Copper', 'Bitcoin', 'Platinum', 'Ethereum', 'S&P_500',
            'Nasdaq_100', 'Apple', 'Tesla', 'Microsoft', 'Silver', 'Google', 'Nvidia',
            'Berkshire', 'Netflix', 'Amazon', 'Meta', 'Gold'
        ]

        # Step 3: Model Training
        logger.info("Step 3: Model Training")
        for stock in stock_names:
            try:
                feature_df = feature_engineering.apply_feature_engineering(stock)
                model_trainer = ModelTraining(df=feature_df, target_column=f"{stock}_Price")
                X, y = model_trainer.prepare_data(stock_name=stock)

                logger.info(f"Starting model training for {stock}")
                best_model, mse, rmse, r2, feature_importances, cross_val_scores = model_trainer.train_model(X, y)

                print(f"\nMetrics for {stock}:")
                print(f" - Mean Squared Error (MSE): {mse}")
                print(f" - Root Mean Squared Error (RMSE): {rmse}")
                print(f" - R² Score: {r2}")

            except ModelTrainingError as me:
                logger.error(f"Model training failed for {stock}: {str(me)}")
                continue

    except (DataLoaderError, FeatureEngineeringError, ModelTrainingError) as e:
        logger.error(f"Process Error: {str(e)}")
        print(f"Process Error: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
