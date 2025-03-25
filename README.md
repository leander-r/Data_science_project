# Stock Analysis Dashboard

A Streamlit-based web application for analyzing stock data, visualizing trends, and predicting future prices using linear regression.

## Overview

This project provides an interactive dashboard to explore stock market data, visualize price trends, trading volumes, correlations, and predict future prices using a simple linear regression model.

## Features

- **Stock Selection**: Select a stock from a sidebar dropdown.
- **Stock Price Trends**: Visualize open, high, low, and close prices over time.
- **Trading Volume**: View trading volume with interactive bar charts.
- **Correlation Heatmap**: Analyze correlations between stock features (open, high, low, close, volume).
- **Price Prediction**: Predict future close prices using linear regression and forecast for the next 30 days.

## Dataset

The project uses the `all_stocks_5yr.csv` dataset, which contains 5 years of stock data with the following columns: `date`, `open`, `high`, `low`, `close`, `volume`, and `Name`.

**Source**: [Kaggle - All Stocks 5yr](https://www.kaggle.com/datasets/rohitjain454/all-stocks-5yr)

**Note**: The dataset is not included in this repository due to its size. Download it from the link above and place it in the `data/` directory.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/leander-r/Data_science_project.git
   cd Data_science_project
