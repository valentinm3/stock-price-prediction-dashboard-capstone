# src/feature_engineering.py
# Author: Valentin Mehedinteanu

import logging

import pandas as pd
from typing import List

from src.logger import Logger



class FeatureEngineeringError(Exception):
    """Custom exception for FeatureEngineering class."""
    pass


class FeatureEngineering:
    """
    A class to perform various feature engineering tasks on a financial dataset.

    Attributes:
        df (pd.DataFrame): DataFrame to engineer features on.
        logger (Logger): Logger instance to handle logging information.
    """

    def __init__(self, df: pd.DataFrame, logger: logging.Logger = None):
        """
        Initializes the FeatureEngineering class with a DataFrame and a logger.

        Args:
            df (pd.DataFrame): The DataFrame to engineer features on.
            logger (logging.Logger, optional): A logger instance. If not provided, a default logger is initialized.
        """
        self.df = df.copy()
        self.logger = logger if logger else Logger.get_logger_for_module(__name__, 'logs/feature_engineering.log')
        self.logger.propagate = False  # Prevent log propagation to the root logger

    def add_moving_average(self, column: str, window: int = 50) -> pd.Series:
        """
        Add Moving Average (MA) feature for a given column.

        Args:
            column (str): The column to calculate the moving average on.
            window (int, optional): The window size for moving average. Defaults to 50.

        Returns:
            pd.Series: The moving average series.

        Raises:
            FeatureEngineeringError: If adding the moving average fails.
        """
        try:
            # Calculate moving average and forward-fill missing values
            ma = self.df[column].rolling(window=window).mean().rename(f'{column}_MA_{window}').ffill()
            return ma
        except Exception as e:
            self._log_and_handle_error(f"Error adding Moving Average for {column}", e)

    def add_exponential_moving_average(self, column: str, window: int = 12) -> pd.Series:
        """
        Add Exponential Moving Average (EMA) feature for a given column.

        Args:
            column (str): The column to calculate the EMA on.
            window (int, optional): The window size for EMA. Defaults to 12.

        Returns:
            pd.Series: The EMA series.

        Raises:
            FeatureEngineeringError: If adding the EMA fails.
        """
        try:
            # Calculate exponential moving average and forward-fill missing values
            ema = self.df[column].ewm(span=window, adjust=False).mean().rename(f'{column}_EMA_{window}').ffill()
            return ema
        except Exception as e:
            self._log_and_handle_error(f"Error adding EMA for {column}", e)

    def add_rsi(self, column: str, window: int = 14) -> pd.Series:
        """
        Add Relative Strength Index (RSI) feature for a given column.

        Args:
            column (str): The column to calculate RSI on.
            window (int, optional): The window size for RSI. Defaults to 14.

        Returns:
            pd.Series: The RSI series.

        Raises:
            FeatureEngineeringError: If adding the RSI fails.
        """
        try:
            # Calculate price changes
            delta = self.df[column].diff()

            # Separate positive gains and negative losses
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            # Calculate rolling average of gains and losses
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()

            # Calculate Relative Strength Index (RSI)
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs)).rename(f'{column}_RSI_{window}').ffill()
            return rsi
        except Exception as e:
            self._log_and_handle_error(f"Error adding RSI for {column}", e)

    def add_rate_of_change(self, column: str, window: int = 10) -> pd.Series:
        """
        Add Rate of Change (ROC) feature for a given column.

        Args:
            column (str): The column to calculate ROC on.
            window (int, optional): The window size for ROC. Defaults to 10.

        Returns:
            pd.Series: The ROC series.

        Raises:
            FeatureEngineeringError: If adding the ROC fails.
        """
        try:
            # Calculate rate of change (percentage change over 'window' periods)
            roc = self.df[column].pct_change(periods=window).mul(100).rename(f'{column}_ROC_{window}').ffill()
            return roc
        except Exception as e:
            self._log_and_handle_error(f"Error adding ROC for {column}", e)

    def add_volatility(self, column: str, window: int = 30) -> pd.Series:
        """
        Add Volatility feature (standard deviation of returns) for a given column.

        Args:
            column (str): The column to calculate volatility on.
            window (int, optional): The window size for volatility calculation. Defaults to 30.

        Returns:
            pd.Series: The volatility series.

        Raises:
            FeatureEngineeringError: If adding the volatility fails.
        """
        try:
            # Calculate rolling standard deviation of percentage change (volatility)
            volatility = self.df[column].pct_change().rolling(window=window).std().rename(
                f'{column}_Volatility_{window}'
            ).ffill()
            return volatility
        except Exception as e:
            self._log_and_handle_error(f"Error adding Volatility for {column}", e)

    def add_lag(self, column: str, lag: int = 1) -> pd.Series:
        """
        Add Lag feature (previous day’s price) for a given column.

        Args:
            column (str): The column to create a lag for.
            lag (int, optional): The number of periods to lag. Defaults to 1.

        Returns:
            pd.Series: The lagged series.

        Raises:
            FeatureEngineeringError: If adding the lag feature fails.
        """
        try:
            # Shift the column values by 'lag' periods to create a lag feature
            lag_feature = self.df[column].shift(lag).rename(f'{column}_Lag_{lag}').ffill()
            return lag_feature
        except Exception as e:
            self._log_and_handle_error(f"Error adding Lag for {column}", e)

    def add_bollinger_bands(self, column: str, window: int = 20, num_std_dev: int = 2) -> pd.DataFrame:
        """
        Add Bollinger Bands (upper and lower) for a given column.

        Args:
            column (str): The column to calculate Bollinger Bands on.
            window (int, optional): The window size for the moving average. Defaults to 20.
            num_std_dev (int, optional): The number of standard deviations for the bands. Defaults to 2.

        Returns:
            pd.DataFrame: A DataFrame containing the upper and lower Bollinger Bands.

        Raises:
            FeatureEngineeringError: If adding the Bollinger Bands fails.
        """
        try:
            # Calculate rolling mean and standard deviation
            rolling_mean = self.df[column].rolling(window=window).mean()
            rolling_std = self.df[column].rolling(window=window).std()

            # Calculate upper and lower Bollinger Bands
            upper_band = (rolling_mean + num_std_dev * rolling_std).rename(f'{column}_BB_Upper')
            lower_band = (rolling_mean - num_std_dev * rolling_std).rename(f'{column}_BB_Lower')

            # Concatenate upper and lower bands
            return pd.concat([upper_band, lower_band], axis=1)
        except Exception as e:
            self._log_and_handle_error(f"Error adding Bollinger Bands for {column}", e)

    def add_time_based_features(self, date_column: str = 'Date') -> pd.DataFrame:
        """
        Add time-based features: days since start, month, day of week, quarter, and day of year.

        Args:
            date_column (str, optional): The name of the date column. Defaults to 'Date'.

        Returns:
            pd.DataFrame: A DataFrame containing the new time-based features.

        Raises:
            FeatureEngineeringError: If adding time-based features fails.
        """
        try:
            # Ensure that the date column exists in the DataFrame
            if date_column not in self.df.columns:
                raise FeatureEngineeringError(f"No '{date_column}' column found in DataFrame.")

            # Convert the date column to datetime
            self.df[date_column] = pd.to_datetime(self.df[date_column])

            # Create new time-based features
            time_features = pd.DataFrame(index=self.df.index)
            time_features['Days_Since_Start'] = (self.df[date_column] - self.df[date_column].min()).dt.days
            time_features['Month'] = self.df[date_column].dt.month
            time_features['Day_of_Week'] = self.df[date_column].dt.dayofweek
            time_features['Quarter'] = self.df[date_column].dt.quarter
            time_features['Day_of_Year'] = self.df[date_column].dt.dayofyear

            self.logger.info('Time-based features added successfully.')
            return time_features
        except Exception as e:
            self._log_and_handle_error('Error adding time-based features', e)

    def apply_feature_engineering(self, stock_name: str) -> pd.DataFrame:
        """
        Apply all feature engineering steps for each given stock.

        Args:
            stock_name (str): The name of the stock to engineer features for.

        Returns:
            pd.DataFrame: DataFrame containing engineered features for the specified stock.

        Raises:
            FeatureEngineeringError: If any step in feature engineering fails.
        """
        try:
            self.logger.info(f'Starting feature engineering for {stock_name}')
            price_column = f'{stock_name}_Price'

            # List of new features created by the various feature engineering methods
            new_features: List[pd.DataFrame] = [
                self.add_moving_average(price_column).to_frame(),
                self.add_exponential_moving_average(price_column).to_frame(),
                self.add_volatility(price_column).to_frame(),
                self.add_lag(price_column, lag=7).to_frame(),
                self.add_lag(price_column, lag=30).to_frame(),
                self.add_bollinger_bands(price_column, window=20, num_std_dev=2),
                self.add_rsi(price_column, window=14),
                self.add_rate_of_change(price_column, window=10),
                self.add_time_based_features(date_column='Date')
            ]

            # Concatenate original DataFrame with new features
            self.df = pd.concat([self.df] + new_features, axis=1)

            self.logger.info(f'Feature engineering completed for {stock_name}')
            return self.df
        except Exception as e:
            self._log_and_handle_error(f"Error applying feature engineering for {stock_name}", e)

    def _log_and_handle_error(self, message: str, exception: Exception) -> None:
        """
        Helper method to log and raise FeatureEngineeringError.

        Args:
            message (str): The message to log before raising the error.
            exception (Exception): The exception to raise.

        Raises:
            FeatureEngineeringError: Logs the error and raises the exception.
        """
        self.logger.error(f"{message}: {str(exception)}")
        raise FeatureEngineeringError(f"{message}: {str(exception)}") from exception
