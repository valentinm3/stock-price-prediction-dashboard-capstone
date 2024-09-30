# src/app.py
# Author: Valentin Mehedinteanu

import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from src.visualizer import Visualizer, VisualizerError
from src.predictive_service import PredictiveService, PredictionError
from src.logger import Logger


class AppError(Exception):
    """Custom exception for app.py errors."""
    pass


def app():
    """
    Main function to set up the Streamlit app for stock price analysis.
    """

    # ---------------------------------------------------------------------
    # Streamlit App Configuration
    # ---------------------------------------------------------------------
    st.set_page_config(
        page_title="📊 Stock Price Prediction Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---------------------------------------------------------------------
    # Add Project Title
    # ---------------------------------------------------------------------
    st.title("Stock Price Prediction Dashboard")

    # ---------------------------------------------------------------------
    # Initialize Logger
    # ---------------------------------------------------------------------
    # Ensure the logs directory exists
    log_directory = 'logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Initialize the logger
    logger = Logger.get_logger_for_module(__name__, os.path.join(log_directory, 'app.log'))

    # ---------------------------------------------------------------------
    # Initialize Visualizer
    # ---------------------------------------------------------------------
    visualizer = Visualizer()

    # ---------------------------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------------------------
    @st.cache_data(show_spinner=False)
    def load_data(data_path: str, _app_logger: logging.Logger) -> pd.DataFrame:
        """
        Load stock data from a CSV file.

        Args:
            data_path (str): Path to the stock data CSV file.
            _app_logger (logging.Logger): Logger instance for logging errors.

        Returns:
            pd.DataFrame: DataFrame containing the stock data.

        Raises:
            AppError: If data loading or parsing fails.
        """
        try:
            load_df = pd.read_csv(data_path)

            # Standardize column names: strip spaces
            load_df.columns = load_df.columns.str.strip()

            _app_logger.info(f"Data loaded successfully from {data_path}")
            return load_df
        except FileNotFoundError:
            st.error(f"Data file not found at path: {data_path}")
            _app_logger.error(f"Data file not found at path: {data_path}")
            return pd.DataFrame()
        except Exception as err:
            st.error(f"Error loading data: {err}")
            _app_logger.error(f"Error loading data: {err}")
            raise AppError(f"Failed to load data from {data_path}: {err}") from err

    # ---------------------------------------------------------------------
    # Initialize PredictiveService
    # ---------------------------------------------------------------------
    try:
        predictive_service = PredictiveService(model_dir='models/')
    except Exception as e:
        logger.error(f"Error initializing PredictiveService: {str(e)}")
        st.error(f"Error initializing predictive service: {str(e)}")
        st.stop()

    # ---------------------------------------------------------------------
    # Load Data
    # ---------------------------------------------------------------------
    CLEANED_DATA_PATH = 'data/cleaned_stock_data.csv'

    try:
        df_cleaned = load_data(CLEANED_DATA_PATH, logger)
    except AppError as ae:
        logger.error(f"AppError occurred while loading data: {str(ae)}")
        st.error(f"AppError occurred while loading data: {str(ae)}")
        st.stop()

    # ---------------------------------------------------------------------
    # Validate Data
    # ---------------------------------------------------------------------
    if df_cleaned.empty:
        logger.error("No cleaned data found. Exiting.")
        st.error("No cleaned data found. Exiting.")
        st.stop()

    # Extract stock names
    price_columns = [col for col in df_cleaned.columns if col.endswith('_Price')]
    stock_names = [col.replace('_Price', '') for col in price_columns]

    if not stock_names:
        logger.error("No stock names found in cleaned data. Exiting.")
        st.error("No stock names found in cleaned data. Exiting.")
        st.stop()

    # ---------------------------------------------------------------------
    # Sidebar - User Inputs
    # ---------------------------------------------------------------------
    st.sidebar.header("🔍 Filter Options")

    # Dropdown for stock selection
    selected_stock = st.sidebar.selectbox("Select a Stock", options=stock_names)

    # ---------------------------------------------------------------------
    # Main Content
    # ---------------------------------------------------------------------
    price_col = f"{selected_stock}_Price"

    if price_col not in df_cleaned.columns:
        st.error(f"Price data for '{selected_stock}' not found in data.")
        logger.error(f"Price data for '{selected_stock}' not found in data.")
        st.stop()

    # ---------------------------------------------------------------------
    # Extract Data for Visuals
    # ---------------------------------------------------------------------
    df_stock = df_cleaned[['Date', price_col]].copy()
    df_stock = df_stock.rename(columns={price_col: 'Price'})
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    df_stock = df_stock.sort_values('Date')

    # Extract latest price information
    latest_price_row = df_stock.iloc[-1]
    latest_date = latest_price_row['Date'].date()
    latest_price = latest_price_row['Price']

    # ---------------------------------------------------------------------
    # Visual 1: Predict Today's Price
    # ---------------------------------------------------------------------
    st.subheader(f"📊 {selected_stock.replace('_', ' ')} - Predict Price 30 Days From Today")

    st.info(f"**Most recent price for {selected_stock.replace('_', ' ')} on {latest_date}:** ${latest_price:,.2f}")

    if st.button("🔮 Predict Price"):
        with st.spinner("Predicting..."):
            try:
                # Get the predicted prices for 30 days into the future
                df_prediction = predictive_service.make_future_predictions(selected_stock, days=30)

                if df_prediction.empty or 'Predicted_Price' not in df_prediction.columns:
                    st.error("Prediction data is unavailable or improperly formatted.")
                    logger.error("Prediction data is unavailable or improperly formatted.")
                else:
                    # Extract the predicted price for the 30th day
                    predicted_price = df_prediction['Predicted_Price'].iloc[-1]

                    # Convert to a float to ensure proper formatting
                    predicted_price = float(predicted_price)

                    # Determine the date 30 days from now
                    date_30_days_from_now = datetime.now().date() + timedelta(days=30)

                    # Display the predicted price
                    st.success(
                        f"**Predicted price for {selected_stock.replace('_', ' ')} on {date_30_days_from_now}:** ${predicted_price:,.2f}"
                    )

            except PredictionError as pe:
                st.error(f"Prediction error: {pe}")
                logger.error(f"Prediction error: {pe}")
            except FileNotFoundError as fnf:
                st.error(f"Model file not found for {selected_stock}: {fnf}")
                logger.error(f"Model file not found for {selected_stock}: {fnf}")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
                logger.error(f"Unexpected error during prediction: {e}")

    # ---------------------------------------------------------------------
    # Visual 2: Model Metrics
    # ---------------------------------------------------------------------
    try:
        # Load model metrics
        metrics_df = pd.read_csv('data/model_training_metrics.csv')

        # Ensure the metrics DataFrame has the necessary columns
        required_columns = ['Stock', 'R2', 'MSE', 'RMSE', 'MAE']
        if not all(col in metrics_df.columns for col in required_columns):
            raise ValueError(f"Metrics DataFrame must contain columns: {required_columns}")

        # Extract metrics for the selected stock
        stock_metrics = metrics_df[metrics_df['Stock'] == selected_stock].iloc[-1]
        st.markdown(
            f"""
            **Model Metrics:**
            - R²: {stock_metrics['R2']:.4f}
            - Mean Squared Error (MSE): {stock_metrics['MSE']:.2f}
            - Root Mean Squared Error (RMSE): {stock_metrics['RMSE']:.2f}
            - Mean Absolute Error (MAE): {stock_metrics['MAE']:.2f}
            """
        )
    except Exception as e:
        st.warning("Metrics data is unavailable or improperly formatted.")
        logger.error(f"Error displaying metrics for {selected_stock}: {e}")

    # ---------------------------------------------------------------------
    # Visual 3: Price over Time (Full History)
    # ---------------------------------------------------------------------
    st.subheader(f"📈 {selected_stock.replace('_', ' ')} - Price over Time (Full History)")

    if df_stock.empty:
        st.warning("No data available for the selected stock.")
    else:
        fig_price = visualizer.plot_price(df_stock, selected_stock)
        if fig_price:
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.warning(f"Could not generate the Price plot for {selected_stock}.")
            logger.error(f"Could not generate Price plot for {selected_stock}")

    # ---------------------------------------------------------------------
    # Visual 4: Feature Importance
    # ---------------------------------------------------------------------
    st.subheader(f"🔍 {selected_stock.replace('_', ' ')} - Feature Importance")

    try:
        model = predictive_service.models.get(selected_stock)

        if not model:
            raise FileNotFoundError(f"Model for {selected_stock} not found in the loaded models.")

        feature_names = predictive_service.get_feature_names(selected_stock)
        logger.info(f"Expected Features: {feature_names}")

        fig_feature_importance = visualizer.plot_feature_importance(model, feature_names, selected_stock)

        if fig_feature_importance:
            st.plotly_chart(fig_feature_importance, use_container_width=True)
        else:
            st.warning("Unable to generate feature importance plot.")
            logger.error("Unable to generate feature importance plot.")
    except PredictionError as pe:
        st.error(f"Prediction error while generating feature importance: {pe}")
        logger.error(f"Prediction error while generating feature importance: {pe}")
    except FileNotFoundError as file_not_found:
        st.error(f"Model file for {selected_stock} not found: {file_not_found}")
        logger.error(f"Model file for {selected_stock} not found: {file_not_found}")
    except Exception as e:
        st.error(f"Unexpected error generating feature importance: {e}")
        logger.error(f"Unexpected error generating feature importance: {e}")

    # ---------------------------------------------------------------------
    # Visual 5: Correlation Heatmap Between Stocks
    # ---------------------------------------------------------------------
    st.subheader("🔥 Correlation Heatmap Between Stocks")

    try:
        selected_stocks = st.multiselect(
            "Select Stocks for Correlation Heatmap",
            options=stock_names,
            default=stock_names[:5]
        )

        if len(selected_stocks) < 2:
            st.warning("Please select at least two stocks to display the correlation heatmap.")
        else:
            # Extract relevant data for the selected stocks
            price_columns = [f"{stock}_Price" for stock in selected_stocks]
            df_corr = df_cleaned[price_columns].copy()
            df_corr.columns = selected_stocks  # Rename for clarity

            # Generate the heatmap using Visualizer
            fig_corr = visualizer.plot_correlation_heatmap(df_cleaned, selected_stocks)

            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Could not generate the correlation heatmap.")
                logger.error("Could not generate correlation heatmap.")
    except VisualizerError as ve:
        st.error(f"Visualizer error while generating correlation heatmap: {ve}")
        logger.error(f"Visualizer error while generating correlation heatmap: {ve}")
    except Exception as e:
        st.error(f"Unexpected error generating correlation heatmap: {e}")
        logger.error(f"Unexpected error generating correlation heatmap: {e}")

    # ---------------------------------------------------------------------
    # Footer
    # ---------------------------------------------------------------------
    st.markdown("---")
    st.markdown("Developed by [Valentin Mehedinteanu] | © 2024")


if __name__ == "__main__":
    app()
