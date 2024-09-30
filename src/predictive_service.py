# src/predictive_service.py
# Author: Valentin Mehedinteanu

import os
import logging
import datetime
from typing import List, Optional

import pandas as pd
import joblib

from src.logger import Logger



class PredictionError(Exception):
    """Custom exception for prediction-related errors."""
    pass


class PredictiveService:
    """
    A service class to manage model predictions for stock prices.

    Attributes:
        model_dir (str): Directory containing trained models.
        predictions_file (Optional[str]): Path to save predictions.
        logger (Logger): Logger instance for recording operations and handling errors.
        models (dict): Loaded trained models for prediction.
    """

    def __init__(self, model_dir: str, predictions_file: Optional[str] = None, logger: logging.Logger = None):
        """
        Initializes the PredictiveService class with the model directory and logger.

        Args:
            model_dir (str): Directory containing trained models.
            predictions_file (Optional[str], optional): File path for saving predictions. Defaults to None.
            logger (logging.Logger, optional): A logger instance. If not provided, a default logger is initialized.
        """
        self.model_dir = model_dir
        self.predictions_file = predictions_file
        self.logger = logger if logger else Logger.get_logger_for_module(__name__, 'logs/predictive_service.log')
        self.logger.propagate = False  # Prevent log propagation to root
        self.models = self.load_models()

    def load_models(self) -> dict:
        """
        Load trained models for all stocks from the specified directory.

        Returns:
            dict: Dictionary mapping stock names to their trained models.

        Raises:
            PredictionError: If there is an error while loading models.
        """
        models = {}
        try:
            for file in os.listdir(self.model_dir):
                if file.endswith('_model.pkl'):
                    stock_name = file.replace('_model.pkl', '')
                    model_path = os.path.join(self.model_dir, file)
                    models[stock_name] = joblib.load(model_path)
                    self.logger.info(f"Loaded model for {stock_name} from {model_path}")
            return models
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise PredictionError(f"Error loading models: {e}") from e

    def get_feature_names(self, stock_name: str) -> List[str]:
        """
        Retrieve feature names for a given stock.

        Args:
            stock_name (str): The name of the stock.

        Returns:
            List[str]: List of feature names.

        Raises:
            PredictionError: If there is an error retrieving feature names.
        """
        feature_file = f"data/features/{stock_name}_features.csv"
        try:
            df_features = pd.read_csv(feature_file)
            feature_names = df_features.columns.tolist()

            # Remove the target price column if it exists
            price_col = f"{stock_name}_Price"
            if price_col in feature_names:
                feature_names.remove(price_col)

            self.logger.info(f"Feature names for {stock_name}: {feature_names}")
            return feature_names
        except Exception as e:
            self.logger.error(f"Error retrieving feature names for {stock_name}: {e}")
            raise PredictionError(f"Error retrieving feature names for {stock_name}: {e}") from e

    def get_latest_features(self, stock_name: str, feature_names: List[str]) -> List[float]:
        """
        Retrieve the latest feature set for the given stock.

        Args:
            stock_name (str): The name of the stock.
            feature_names (List[str]): List of feature names.

        Returns:
            List[float]: List of feature values.

        Raises:
            PredictionError: If there is an error retrieving the latest features.
        """
        feature_file = f"data/features/{stock_name}_features.csv"
        try:
            df_features = pd.read_csv(feature_file)

            # Ensure all required features are present in the DataFrame
            missing_features = [feature for feature in feature_names if feature not in df_features.columns]
            if missing_features:
                raise PredictionError(f"Missing features in the data: {missing_features}")

            # Extract the latest features
            latest_features = df_features.iloc[-1][feature_names].values.tolist()
            self.logger.info(f"Latest features for {stock_name}: {latest_features}")
            return latest_features
        except Exception as e:
            self.logger.error(f"Error retrieving latest features for {stock_name}: {e}")
            raise PredictionError(f"Error retrieving latest features for {stock_name}: {e}") from e

    def make_future_predictions(self, stock_name: str, days: int = 1) -> pd.DataFrame:
        """
        Predict future prices for a given stock.

        Args:
            stock_name (str): The name of the stock to predict.
            days (int): Number of days to predict.

        Returns:
            pd.DataFrame: DataFrame containing 'Date' and 'Predicted_Price'.

        Raises:
            PredictionError: If there is an error making predictions.
        """
        try:
            if stock_name not in self.models:
                raise PredictionError(f"No model found for stock: {stock_name}")

            model = self.models[stock_name]
            feature_names = self.get_feature_names(stock_name)

            # Prepare the input features for prediction
            latest_features = self.get_latest_features(stock_name, feature_names)

            predictions = []
            current_date = datetime.date.today()

            # Loop to predict future prices for the specified number of days
            for day in range(days):
                # Convert features to DataFrame with correct column names
                latest_features_df = pd.DataFrame([latest_features], columns=feature_names)

                # Make the prediction
                predicted_price = model.predict(latest_features_df)[0]

                # Append the prediction to the list
                predictions.append({
                    'Date': current_date,
                    'Predicted_Price': float(predicted_price)
                })

                # Update features for the next prediction day
                latest_features = self.update_features_for_next_day(latest_features, predicted_price, stock_name)

                # Increment the date for the next prediction
                current_date += datetime.timedelta(days=1)

            return pd.DataFrame(predictions)
        except Exception as e:
            self.logger.error(f"Error making predictions for {stock_name}: {e}")
            raise PredictionError(f"Error making predictions for {stock_name}: {e}") from e

    def update_features_for_next_day(self, features: List[float], predicted_price: float, stock_name: str) -> List[float]:
        """
        Update the feature set for the next prediction day based on the predicted price.

        Args:
            features (List[float]): The current set of features.
            predicted_price (float): The predicted price for the current day.
            stock_name (str): The name of the stock to retrieve feature names.

        Returns:
            List[float]: Updated features for the next day.
        """
        # Make a copy of the current feature set to update
        updated_features = features.copy()

        # Retrieve feature names to determine which features to update
        feature_names = self.get_feature_names(stock_name)

        # Update lagged price features with the predicted price
        if 'Lagged_Price' in feature_names:
            updated_features[0] = predicted_price

        return updated_features
