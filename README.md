# Data_science_project

# Stock Analysis Dashboard

A Streamlit-based web application for analyzing stock data, visualizing trends, and predicting future prices using linear regression.

## Features
- **Stock Selection**: Choose from a list of stocks via a sidebar.
- **Stock Price Trends**: Visualize open, high, low, and close prices over time.
- **Trading Volume**: Display trading volume with interactive bar charts.
- **Correlation Heatmap**: Analyze correlations between stock features (open, high, low, close, volume).
- **Price Prediction**: Predict future close prices using linear regression and forecast the next 30 days.

## Dataset
The project uses the `all_stocks_5yr.csv` dataset, which contains 5 years of stock data with columns: `date`, `open`, `high`, `low`, `close`, `volume`, and `Name`.

## Requirements
Install the required dependencies using:
```bash
pip install -r requirements.txt
