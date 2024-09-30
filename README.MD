# 📊 Stock Price Prediction Dashboard

## Description
The Stock Price Prediction Dashboard is a web application designed to predict and analyze trends in the US stock market. The project utilizes historical stock data from 2020-2024, leveraging machine learning techniques (Random Forest) to forecast future stock prices. The goal of this dashboard is to help users optimize their investment strategies, reduce financial risk, and gain insights into various stocks through interactive visualizations.

The application offers features such as future price prediction, feature importance analysis, and visualization of model metrics, all presented in an easy-to-use web interface powered by Streamlit.

## Features
- **Stock Price Prediction**: Predict future stock prices using a Random Forest model.
- **Historical Price Visualization**: Plot stock prices over time for selected stocks.
- **Feature Importance Analysis**: Understand which features influence the Random Forest model's predictions the most.
- **Model Evaluation Metrics**: Display metrics like R², Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
- **Correlation Heatmap**: Visualize correlations between different stocks for deeper insights.

## Technologies Used
- **Programming Language**: Python
- **Web Framework**: Streamlit
- **Machine Learning**: Scikit-learn, RandomForestRegressor
- **Data Handling**: Pandas, NumPy
- **Visualization**: Plotly

## Project Structure

- **`data/`**: Contains stock data and metrics files.
- **`models/`**: Stores trained models in `.pkl` format.
- **`logs/`**: Logs for each component to assist with debugging.
- **`src/`**: Source code modules for data handling, feature engineering, model training, prediction, and visualization.
- **`app.py`**: The main application for the web dashboard (Streamlit app).
- **`main.py`**: Runs data preparation, feature engineering, and model training.
- **`requirements.txt`**: Lists dependencies required to run the project.

## Installation

### Prerequisites
Ensure that you have **Python 3.8+** and **pip** installed on your system.

### Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/valentinm3/stock-price-prediction-dashboard-capstone.git
    ```
2. **Navigate to the project directory**:
    ```sh
    cd stock-price-prediction-dashboard-capstone
    ```
3. **Create a Virtual Environment**:
    ```sh
    python -m venv .venv
    ```
4. **Activate the Virtual Environment**:
   - On **Windows**:
     ```sh
     .venv\Scripts\activate
     ```
   - On **Mac/Linux**:
     ```sh
     source .venv/bin/activate
     ```
5. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
6. **Run the app**:
    ```sh
    streamlit run app.py
    ```

## Usage
Once the application is running, you can access the dashboard through your web browser. The dashboard allows you to:

- Select a stock from the available options to view its historical price trends.
- Predict future stock prices based on the selected stock's historical data.
- View feature importance to understand which factors most influence the predictions.
- Analyze model performance through various evaluation metrics.
- Explore correlations between different stocks using the heatmap visualization.

## Contributing
This project was developed solely by Valentin Mehedinteanu as part of a learning endeavor. Contributions are not 
expected or solicited, as the project is intended to be a personal demonstration of the skills and techniques learned.

## License
This project is intended for educational purposes only and is not formally licensed for distribution or commercial use. 
It is part of a personal portfolio to showcase skills in data analysis, machine learning, and web application 
development. Please provide appropriate credit if you decide to use or reference the work.

## Contact
For any questions or inquiries, please contact [valentinm3@gmail.com](mailto:valentinm3@gmail.com).
