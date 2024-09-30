# src/model_training.py
# Author: Valentin Mehedinteanu

import os
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Dict, Any
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer

from src.logger import Logger

warnings.filterwarnings("ignore")


class ModelTrainingError(Exception):
    """Custom exception for ModelTraining class."""
    pass


class ModelTraining:
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        model_dir: str = 'models/',
        logger: logging.Logger = None
    ):
        """
        Initializes the ModelTraining class with data, target column, and model directory.

        Args:
            df (pd.DataFrame): The DataFrame containing features and target.
            target_column (str): The name of the target column to predict.
            model_dir (str, optional): Directory to save trained models. Defaults to 'models/'.
            logger (logging.Logger, optional): A logger instance. If not provided, a default logger is initialized.
        """
        self.df = df.copy()
        self.target_column = target_column
        self.model_dir = model_dir

        # Create the model directory if it does not exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Initialize logger if not provided
        self.logger = logger if logger else Logger.get_logger_for_module(__name__, 'logs/model_training.log')
        self.logger.propagate = False

    def prepare_data(self, stock_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data with relevant features, volume, and original price column.

        Args:
            stock_name (str): The name of the stock to prepare data for.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing X (features) and y (target).

        Raises:
            ModelTrainingError: If an error occurs during data preparation.
        """
        try:
            self.logger.info(f'Preparing data for {stock_name}')

            # Identify the price column
            price_column = f"{stock_name}_Price"
            if price_column not in self.df.columns:
                raise ValueError(f"Price column '{price_column}' not found in DataFrame.")

            # Select feature columns related to the specific stock
            feature_columns = [col for col in self.df.columns if col.startswith(stock_name) and col != price_column]

            # Include price columns for other stocks, excluding volume data
            other_stock_price_columns = [
                col for col in self.df.columns if col.endswith('_Price') and col != price_column
            ]

            # Handle missing values and drop initial rows with NaNs caused by feature engineering
            self.drop_initial_rows(50)
            self.handle_missing_values('ffill')
            self.cap_large_values()

            # Combine features and the target price column
            stock_columns = feature_columns + other_stock_price_columns + [price_column]

            # Create a DataFrame with selected columns
            stock_df = self.df[stock_columns]

            # Remove the 'Date' column if it exists, as it's not required for training
            if 'Date' in stock_df.columns:
                stock_df = stock_df.drop(columns=['Date'])

            # Define features (X) and target (y)
            X = stock_df.drop(columns=[price_column])
            y = stock_df[price_column].squeeze()

            self.logger.info(f'Data prepared successfully for {stock_name}')
            return X, y
        except Exception as e:
            self.logger.error(f'Error preparing data for {stock_name}: {str(e)}')
            raise ModelTrainingError(f'Error preparing data for {stock_name}: {str(e)}') from e

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[
        RandomForestRegressor, float, float, float, pd.Series, Dict[str, float]
    ]:
        """
        Train a Random Forest model with GridSearchCV and TimeSeriesSplit, and evaluate using multiple metrics.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target labels.

        Returns:
            Tuple[RandomForestRegressor, float, float, float, pd.Series, Dict[str, float]]:
                best_model, mse, rmse, r2, feature_importances, cross_val_scores.

        Raises:
            ModelTrainingError: If an error occurs during model training.
        """
        try:
            # Define custom RMSE scoring function for evaluation
            def rmse(y_true, y_prediction):
                return np.sqrt(mean_squared_error(y_true, y_prediction))

            rmse_scorer = make_scorer(rmse, greater_is_better=False)

            # Define scoring metrics
            scoring_metrics = {
                'R2': 'r2',
                'MSE': 'neg_mean_squared_error',
                'MAE': 'neg_mean_absolute_error',
                'RMSE': rmse_scorer
            }

            # Define parameter grid for Random Forest optimization
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True, False],
                'max_features': [0.5, 0.7]
            }

            rf = RandomForestRegressor(random_state=42)
            tscv = TimeSeriesSplit(n_splits=3)

            use_random_search = True

            # Use RandomizedSearchCV or GridSearchCV based on `use_random_search`
            if use_random_search:
                n_iter_search = 100
                search = RandomizedSearchCV(
                    estimator=rf,
                    param_distributions=param_grid,
                    n_iter=n_iter_search,
                    cv=tscv,
                    scoring=scoring_metrics,
                    n_jobs=-2,
                    random_state=42,
                    verbose=1,
                    refit='RMSE'
                )
            else:
                search = GridSearchCV(
                    estimator=rf,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring=scoring_metrics,
                    n_jobs=-2,
                    verbose=1,
                    refit='RMSE'
                )

            # Train the model using cross-validation
            search.fit(X, y)

            # Extract the best model and parameters
            best_model = search.best_estimator_
            best_params = search.best_params_
            self.logger.info(f"Best parameters for {self.target_column}: {best_params}")

            # Evaluate the best model using cross-validation scores
            cv_scores = cross_validate(
                best_model,
                X,
                y,
                scoring=scoring_metrics,
                cv=tscv,
                n_jobs=-2,
                return_train_score=False
            )

            # Organize cross-validation scores
            cross_val_scores = {
                'R2': cv_scores['test_R2'].mean(),
                'MSE': -cv_scores['test_MSE'].mean(),
                'MAE': -cv_scores['test_MAE'].mean(),
                'RMSE': np.sqrt(-cv_scores['test_MSE'].mean())
            }

            # Extract feature importances and sort them
            feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
            feature_importances_sorted = feature_importances.sort_values(ascending=False)

            # Save the model and feature-engineered data
            self.save_model(best_model)
            self.save_feature_engineered_data_per_stock()
            self.save_metrics_to_csv(best_params, cross_val_scores)

            return best_model, cross_val_scores['MSE'], cross_val_scores['RMSE'], cross_val_scores['R2'], feature_importances_sorted, cross_val_scores
        except Exception as e:
            self.logger.error(f'Error training model for {self.target_column}: {str(e)}')
            raise ModelTrainingError(f'Error training model for {self.target_column}: {str(e)}') from e

    def save_feature_engineered_data_per_stock(self) -> None:
        """
        Save feature-engineered data for the specific stock to a CSV file.

        Raises:
            ModelTrainingError: If saving the data fails.
        """
        try:
            if self.df.empty:
                self.logger.warning('No data to save. Perform feature engineering first.')
                raise ModelTrainingError('No data to save.')

            target_stock_name = self.target_column.replace('_Price', '')
            stock_columns = [
                col for col in self.df.columns if col.startswith(target_stock_name) and col != self.target_column
            ]

            other_stock_price_columns = [
                col for col in self.df.columns if col.endswith('_Price') and col != self.target_column
            ]

            relevant_columns = stock_columns + other_stock_price_columns
            stock_df = self.df[relevant_columns]

            output_dir = 'data/features'
            os.makedirs(output_dir, exist_ok=True)

            stock_file_path = os.path.join(output_dir, f'{target_stock_name}_features.csv')
            stock_df.to_csv(stock_file_path, index=False)
            self.logger.info(f'Feature-engineered data saved to {stock_file_path}')
        except Exception as e:
            self.logger.error(f'Error saving feature-engineered data for {self.target_column}: {str(e)}')
            raise ModelTrainingError(f'Error saving feature-engineered data for {self.target_column}: {str(e)}') from e

    def save_model(self, model: RandomForestRegressor) -> None:
        """
        Save the trained model to the model directory.

        Args:
            model (RandomForestRegressor): The trained model to save.

        Raises:
            ModelTrainingError: If an error occurs while saving the model.
        """
        try:
            model_filename = f"{self.target_column.replace('_Price', '')}_model.pkl"
            model_path = os.path.join(self.model_dir, model_filename)
            joblib.dump(model, model_path)
            self.logger.info(f'Model saved successfully to {model_path}')
        except Exception as e:
            self.logger.error(f'Error saving model for {self.target_column}: {str(e)}')
            raise ModelTrainingError(f'Error saving model for {self.target_column}: {str(e)}') from e

    def save_metrics_to_csv(self, best_params: Dict[str, Any], cross_val_scores: Dict[str, float]) -> None:
        """
        Save model training metrics to a CSV file with the current date, parameters, and metrics.

        Args:
            best_params (Dict[str, Any]): The best parameters from GridSearchCV or RandomizedSearchCV.
            cross_val_scores (Dict[str, float]): The cross-validation scores.

        Returns:
            None
        """
        try:
            metrics_file_path = 'data/model_training_metrics.csv'
            current_date = datetime.now().date()
            search_type = "RandomizedSearchCV" if 'n_iter' in best_params else "GridSearchCV"

            metrics_data = {
                'Date': [current_date],
                'Stock': [self.target_column.replace('_Price', '')],
                'Search Type': [search_type],
                'Best Parameters': [best_params],
                'R2': [cross_val_scores['R2']],
                'MSE': [cross_val_scores['MSE']],
                'RMSE': [cross_val_scores['RMSE']],
                'MAE': [cross_val_scores['MAE']]
            }

            metrics_df = pd.DataFrame(metrics_data)

            # Append metrics to the existing CSV or create a new file if not found
            if os.path.exists(metrics_file_path):
                if os.path.getsize(metrics_file_path) > 0:
                    metrics_df.to_csv(metrics_file_path, mode='a', header=False, index=False)
                else:
                    metrics_df.to_csv(metrics_file_path, mode='w', header=True, index=False)
            else:
                metrics_df.to_csv(metrics_file_path, mode='w', header=True, index=False)

            self.logger.info(f'Metrics saved successfully for {self.target_column}')
        except Exception as e:
            self.logger.error(f'Error saving metrics for {self.target_column}: {str(e)}')

    def handle_missing_values(self, method: str = 'ffill') -> None:
        """
        Handle missing values and infinite values in the DataFrame after feature engineering.

        Args:
            method (str, optional): The method to handle missing values.
                                    Options are 'ffill', 'bfill', or 'drop'. Defaults to 'ffill'.

        Raises:
            FeatureEngineeringError: If handling missing values fails.
        """
        try:
            self.logger.info(f'Handling missing and infinite values using method: {method}')

            # Replace infinite values with NaN
            self.df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)

            # Handle NaN values according to the method
            if method == 'ffill':
                self.df.ffill(inplace=True)
            elif method == 'bfill':
                self.df.bfill(inplace=True)
            elif method == 'drop':
                self.df.dropna(inplace=True)
            else:
                raise ValueError(f"Unsupported method for handling missing values: {method}")

            # After handling, check for any remaining NaN or infinite values
            if self.df.isna().sum().sum() > 0:
                raise ModelTrainingError("NaN values still exist after handling.")

            self.logger.info('Missing and infinite values handled successfully.')
        except Exception as e:
            self.logger.error('Error handling missing or infinite values', e)

    def cap_large_values(self) -> None:
        """
        Cap both large and small values in each numeric column at the specified percentiles.

        Raises:
            FeatureEngineeringError: If capping fails.
        """
        try:
            # Capping for price columns (1st and 99th percentile)
            price_columns = [
                'Natural_Gas_Price', 'Crude_oil_Price', 'S&P_500_Price', 'Meta_Price', 'Gold_Price',
                'Silver_Price', 'Nvidia_Price'
            ]
            for col in price_columns:
                lower_limit = self.df[col].quantile(0.01)
                upper_limit = self.df[col].quantile(0.99)
                self.df[col] = self.df[col].clip(lower=lower_limit, upper=upper_limit)

            # Capping for volume columns (5th and 95th percentile)
            volume_columns = [col for col in self.df.columns if '_Vol.' in col]
            for col in volume_columns:
                lower_limit = self.df[col].quantile(0.05)
                upper_limit = self.df[col].quantile(0.95)
                self.df[col] = self.df[col].clip(lower=lower_limit, upper=upper_limit)

            self.logger.info("Successfully capped values.")
        except Exception as e:
            self.logger.error("Error capping values", e)

    def drop_initial_rows(self, n: int) -> None:
        """
        Drop the first 'n' rows from the DataFrame to eliminate NaN values introduced by rolling calculations.

        Args:
            n (int): Number of rows to drop.

        Raises:
            FeatureEngineeringError: If dropping rows results in an empty DataFrame.
        """
        try:
            if n >= len(self.df):
                raise ModelTrainingError(f"Cannot drop {n} rows from DataFrame with {len(self.df)} rows.")

            self.logger.info(f"Dropping the first {n} rows from the DataFrame.")
            self.df = self.df.iloc[n:].reset_index(drop=True)
            self.logger.info(f"Dropped the first {n} rows successfully.")
        except Exception as e:
            self.logger.error(f"Error dropping initial rows: {str(e)}")
            raise ModelTrainingError(f"Error dropping initial rows: {str(e)}") from e
