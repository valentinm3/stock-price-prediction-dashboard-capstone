# src/visualizer.py
# Author: Valentin Mehedinteanu

import logging
from typing import Optional, List

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from src.logger import Logger



class VisualizerError(Exception):
    """Custom exception for Visualizer class."""
    pass


class Visualizer:
    """
    A class for generating visualizations related to financial data analysis.

    Attributes:
        logger (Logger): Logger instance for recording operations and handling errors.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initializes the Visualizer class with a logger.

        Args:
            logger (Logger, optional): A logger instance. If not provided, a default logger is initialized.
        """
        # Initialize logger using the new utility method
        self.logger = logger if logger else Logger.get_logger_for_module(__name__, 'logs/visualizer.log')
        self.logger.propagate = False  # Prevent log propagation to root

    def plot_future_predictions(self, df_historical: pd.DataFrame, df_predictions: pd.DataFrame, stock_name: str) -> Optional[go.Figure]:
        """
        Plot historical and future predicted prices for a given stock.

        Args:
            df_historical (pd.DataFrame): DataFrame containing historical 'Date' and 'Price'.
            df_predictions (pd.DataFrame): DataFrame containing future 'Date' and 'Predicted_Price'.
            stock_name (str): The name of the stock being plotted.

        Returns:
            plotly.graph_objs.Figure: The generated figure for historical and predicted prices, or None on error.
        """
        try:
            self.logger.info(f"Generating future predictions plot for {stock_name}")
            self.logger.debug(f"Historical data head:\n{df_historical.head()}")
            self.logger.debug(f"Predictions data head:\n{df_predictions.head()}")

            # Create the plotly figure for the historical and predicted prices
            fig = go.Figure()

            # Add predicted prices trace
            fig.add_trace(go.Scatter(
                x=df_predictions['Date'],
                y=df_predictions['Predicted_Price'],
                mode='lines',
                name='Predicted Price',
                line=dict(color='orange', width=2, dash='dash')
            ))

            # Update figure layout to make it more informative
            fig.update_layout(
                title=f"{stock_name} - Historical and Predicted Prices",
                xaxis=dict(title='Date'),
                yaxis=dict(title='Price'),
                legend=dict(x=0, y=1),
                hovermode="x unified"
            )

            self.logger.info(f"Successfully generated future predictions plot for {stock_name}")
            return fig
        except Exception as e:
            self.logger.error(f"Error generating future predictions plot for {stock_name}: {e}")
            raise VisualizerError(f"Error generating future predictions plot for {stock_name}: {e}") from e

    def plot_price(self, df: pd.DataFrame, stock_name: str) -> Optional[go.Figure]:
        """
        Plot the price over time for a given stock.

        Args:
            df (pd.DataFrame): DataFrame containing 'Price'.
            stock_name (str): The name of the stock being plotted.

        Returns:
            plotly.graph_objs.Figure: The generated figure for price over time, or None on error.
        """
        try:
            self.logger.info(f"Generating price plot for {stock_name}")
            self.logger.debug(f"Data for {stock_name}: {df.head()}")

            # Validate required columns are present
            if 'Price' not in df.columns:
                self.logger.error(f"No 'Price' column found in DataFrame for {stock_name}.")
                raise VisualizerError(f"No 'Price' column found in DataFrame for {stock_name}.")
            if 'Date' not in df.columns:
                self.logger.error(f"No 'Date' column found in DataFrame for {stock_name}.")
                raise VisualizerError(f"No 'Date' column found in DataFrame for {stock_name}.")

            # Create plotly figure for price over time
            fig = go.Figure()

            # Add price trace
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Price'],
                mode='lines',
                name=f"{stock_name} Price",
                line=dict(color='royalblue', width=3),
                fill='tozeroy',
                opacity=0.6
            ))

            # Update figure layout
            fig.update_layout(
                title={
                    'text': f"{stock_name.replace('_', ' ')} - Price Over Time",
                    'y': 1,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24)
                },
                xaxis=dict(
                    title='Date',
                    titlefont=dict(size=18),
                    tickformat='%b %d, %Y',
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    tickangle=-45
                ),
                yaxis=dict(
                    title='Price ($)',
                    titlefont=dict(size=18),
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    zeroline=True,
                    zerolinecolor='gray',
                    rangemode='nonnegative'
                ),
                template='plotly_white',
                hovermode="x unified",
                legend=dict(
                    x=0.05,
                    y=0.95,
                    bgcolor='rgba(255, 255, 255, 0.6)',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=14)
                ),
                margin=dict(l=40, r=30, t=70, b=40)
            )

            self.logger.info(f"Successfully generated price plot for {stock_name}")
            return fig
        except Exception as e:
            self.logger.error(f"Error generating price plot for {stock_name}: {e}")
            raise VisualizerError(f"Error generating price plot for {stock_name}: {e}") from e

    def plot_feature_importance(self, model, feature_names: List[str], stock_name: str) -> Optional[go.Figure]:
        """
        Plot the top feature importances for a given stock based on the trained model.

        Args:
            model: The trained model (Random Forest).
            feature_names (list): The names of the features used for training.
            stock_name (str): The name of the stock being plotted.

        Returns:
            plotly.graph_objs.Figure: The generated figure for feature importance, or None on error.
        """
        try:
            self.logger.info(f"Generating feature importance plot for {stock_name}")

            # Get feature importances and sort them in descending order
            importances = model.feature_importances_
            sorted_idx = importances.argsort()[::-1]

            top_n = 10  # Top n features to display
            top_features = sorted_idx[:top_n]

            # Create plotly figure for feature importance
            fig = go.Figure()

            # Add bar trace for top features
            fig.add_trace(go.Bar(
                x=[feature_names[i] for i in top_features],
                y=[importances[i] for i in top_features],
                orientation='v',
                name=f"{stock_name} Feature Importance",
                marker=dict(color='rgba(55, 128, 191, 0.7)'),
            ))

            # Update figure layout
            fig.update_layout(
                title=f"{stock_name} - Top {top_n} Feature Importance",
                xaxis=dict(title='Feature'),
                yaxis=dict(title='Importance'),
                hovermode="x unified"
            )

            self.logger.info(f"Successfully generated feature importance plot for {stock_name}")
            return fig
        except Exception as e:
            self.logger.error(f"Error generating feature importance plot for {stock_name}: {e}")
            raise VisualizerError(f"Error generating feature importance plot for {stock_name}: {e}") from e

    def plot_correlation_heatmap(self, df: pd.DataFrame, stock_names: List[str]) -> Optional[go.Figure]:
        """
        Plot a heatmap showing the correlation between different stocks.

        Args:
            df (pd.DataFrame): DataFrame containing stock prices with columns like 'Stock1_Price', 'Stock2_Price', etc.
            stock_names (List[str]): List of stock names to include in the correlation.

        Returns:
            plotly.graph_objs.Figure: The generated heatmap figure, or None on error.
        """
        try:
            self.logger.info("Generating correlation heatmap for selected stocks.")

            # Extract price columns for the selected stocks
            price_columns = [f"{stock}_Price" for stock in stock_names]
            df_prices = df[price_columns].copy()

            # Rename columns to stock names for clarity
            df_prices.columns = stock_names

            # Compute the correlation matrix
            corr_matrix = df_prices.corr()
            self.logger.debug(f"Correlation matrix:\n{corr_matrix}")

            # Create heatmap using Plotly Express
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Viridis',
                title="Correlation Heatmap of Stock Prices"
            )

            self.logger.info("Successfully generated correlation heatmap.")
            return fig
        except Exception as e:
            self.logger.error(f"Error generating correlation heatmap: {e}")
            raise VisualizerError(f"Error generating correlation heatmap: {e}") from e
