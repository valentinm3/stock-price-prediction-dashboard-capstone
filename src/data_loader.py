# src/data_loader.py
# Author: Valentin Mehedinteanu

import os
import logging

import pandas as pd

from src.logger import Logger



class DataLoaderError(Exception):
    """Custom exception for DataLoader class."""
    pass


class DataLoader:
    def __init__(self, file_path: str, logger: logging.Logger = None):
        """
        Initializes the DataLoader with the specified file path and logger.

        Args:
            file_path (str): The path to the CSV file containing stock data.
            logger (logging.Logger, optional): A logger instance. If not provided, a default logger is initialized.
        """
        self.file_path = file_path
        self.cleaned_file_path = 'data/cleaned_stock_data.csv'
        self.df = None

        # Initialize logger using the provided or default logger
        self.logger = logger if logger else Logger.get_logger_for_module(__name__, 'logs/data_loader.log')
        self.logger.propagate = False  # Prevent log propagation to root

    def load_raw_data(self) -> None:
        """
        Load the CSV file into a DataFrame.

        Raises:
            DataLoaderError: If an error occurs while loading data.
        """
        try:
            self.logger.info(f'Loading data from {self.file_path}')
            self.df = pd.read_csv(self.file_path)
            self.logger.info('Data loaded successfully')
        except Exception as e:
            self.logger.error(f'Error loading data from {self.file_path}: {str(e)}')
            raise DataLoaderError(f'Error loading data from {self.file_path}: {str(e)}') from e

    def clean_data(self) -> None:
        """
        Clean the dataset by fixing date formats and filling missing values.

        Raises:
            DataLoaderError: If no data is loaded or if data cleaning fails.
        """
        if self.df is None:
            self.logger.warning('No data to clean. Please load data first.')
            raise DataLoaderError('No data to clean.')

        try:
            self.logger.info('Cleaning data...')

            # Drop 'Unnamed: 0' column if it exists
            if 'Unnamed: 0' in self.df.columns:
                self.df.drop("Unnamed: 0", axis=1, inplace=True)
                self.logger.info("'Unnamed: 0' column found and dropped")

            # Forward and backward fill for all columns except 'Platinum_Vol.'
            non_platinum_vol_columns = self.df.columns != 'Platinum_Vol.'
            self.df.loc[:, non_platinum_vol_columns] = self.df.loc[:, non_platinum_vol_columns].ffill().bfill()

            # Fill missing values in 'Platinum_Vol.' with 0
            if 'Platinum_Vol.' in self.df.columns:
                self.logger.info('Handling Platinum_Vol. column')
                self.df['Platinum_Vol.'] = self.df['Platinum_Vol.'].fillna(0)
                self.logger.info('Filled NaN values in Platinum_Vol.')

            # Convert 'Date' column to datetime
            self.logger.info('Converting Date column to datetime')
            self.df['Date'] = self.df['Date'].str.replace('/', '-')
            self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%m-%Y')
            self.logger.info('Date column converted successfully')

            # Convert price and volume columns to float64
            self.logger.info('Converting Price and Volume columns to float64')
            price_vol_cols = self.df.filter(regex='Price|Vol.').columns.tolist()
            self.df[price_vol_cols] = self.df[price_vol_cols].replace({',': ''}, regex=True)
            self.df[price_vol_cols] = self.df[price_vol_cols].astype('float64')

            self.logger.info('Data cleaning complete')
        except Exception as e:
            self.logger.error(f'Error cleaning data: {str(e)}')
            raise DataLoaderError(f'Error cleaning data: {str(e)}') from e

    def save_cleaned_data(self) -> None:
        """
        Save the cleaned data to a CSV file.

        Raises:
            DataLoaderError: If no data is available to save or if saving fails.
        """
        try:
            if self.df is None:
                self.logger.warning('No data to save. Please clean data first.')
                raise DataLoaderError('No data to save.')

            self.df.to_csv(self.cleaned_file_path, index=False)
            self.logger.info(f'Cleaned data saved to {self.cleaned_file_path}')
        except Exception as e:
            self.logger.error(f'Error saving cleaned data to {self.cleaned_file_path}: {str(e)}')
            raise DataLoaderError(f'Error saving cleaned data: {str(e)}') from e

    def load_cleaned_data(self) -> pd.DataFrame:
        """
        Return the cleaned dataset from memory or from 'cleaned_stock_data.csv' if available.

        Returns:
            pd.DataFrame: The cleaned data.

        Raises:
            DataLoaderError: If cleaned data cannot be retrieved.
        """
        try:
            # Load cleaned data from file if it exists
            if os.path.exists(self.cleaned_file_path):
                self.logger.info(f"Loading cleaned data from {self.cleaned_file_path}")
                self.df = pd.read_csv(self.cleaned_file_path, parse_dates=['Date'])
                self.logger.info(f'Date column dtype after loading: {self.df["Date"].dtypes}')
                self.logger.info('Cleaned data loaded successfully from file')
                return self.df

            # Return cleaned data from memory if available
            if self.df is not None:
                self.logger.info('Returning cleaned data from memory')
                return self.df

            # If no data is available, raise an error
            self.logger.warning('Data not loaded yet. Please load and clean the data first.')
            raise DataLoaderError('Data not loaded or cleaned.')
        except Exception as e:
            self.logger.error(f"Error retrieving cleaned data from {self.cleaned_file_path}: {str(e)}")
            raise DataLoaderError(f"Error retrieving cleaned data: {str(e)}") from e
