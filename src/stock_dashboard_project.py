import os
import kaggle
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler



# ===========================
#  Secure Kaggle API Handling
# ===========================
# Check if running on Streamlit Cloud (secrets available)

if "KAGGLE_USERNAME" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
elif os.path.exists("kaggle.json"):
    # Load Kaggle API key from local file (for local testing)
    os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()
else:
    st.error("Kaggle API key not found! Please add it to Streamlit Secrets.")

# Kaggle dataset reference
dataset = "rohitjain454/all-stocks-5yr"

kaggle.api.dataset_download_files(dataset, path="../data/", unzip=True)
csv_path = "../data/all_stocks_5yr.csv"

# Load Data
df = pd.read_csv(csv_path)
df["date"] = pd.to_datetime(df["date"])

# Sidebar for stock selection
st.sidebar.title("📌 Stock Selection")
stocks = df["Name"].unique()
selected_stock = st.sidebar.selectbox("Select a Stock:", stocks)

# Filter data for selected stock
stock_data = df[df["Name"] == selected_stock]

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock Prices", "📊 Trading Volume", "🔬 Correlation Heatmap", "📉 Price Prediction"])

# Tab 1: Stock Price Trends
with tab1:
    st.subheader(f"📈 {selected_stock} Stock Price Trends")
    fig = px.line(stock_data, x="date", y=["open", "high", "low", "close"], 
                  labels={"value": "Stock Price", "date": "Date"}, 
                  title=f"{selected_stock} Stock Price Over Time")
    fig.update_layout(legend_title_text="Price Type", xaxis_title="Date", yaxis_title="Stock Price")
    st.plotly_chart(fig)

# Tab 2: Trading Volume
with tab2:
    st.subheader(f"📊 Trading Volume for {selected_stock}")
    fig_vol = px.bar(stock_data, x="date", y="volume", 
                     labels={"volume": "Trading Volume", "date": "Date"}, 
                     title=f"{selected_stock} Trading Volume Over Time",
                     color_discrete_sequence=["#1f77b4"])
    fig_vol.update_layout(xaxis_title="Date", yaxis_title="Volume")
    st.plotly_chart(fig_vol)

# Tab 3: Correlation Heatmap
with tab3:
    st.subheader(f"🔬 Correlation Heatmap for {selected_stock}")
    corr_matrix = stock_data[["open", "high", "low", "close", "volume"]].corr()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Tab 4: Price Prediction using Regression
with tab4:
    st.subheader(f"📉 Stock Price Prediction with Linear Regression for {selected_stock}")
    
    # Preprocessing: Select features and target variable
    features = stock_data[["open", "high", "low", "volume"]]
    target = stock_data["close"]
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

    # Standardize the features (important for regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate Mean Squared Error (MSE) for model performance evaluation
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")

    # Plot Actual vs Predicted prices
    st.subheader(f"Actual vs Predicted Close Price for {selected_stock}")
    fig_pred = px.line(x=y_test.index, y=[y_test, y_pred], 
                       labels={"x": "Index", "y": "Price"}, 
                       title=f"{selected_stock} Actual vs Predicted Close Price")
    fig_pred.update_layout(xaxis_title="Index", yaxis_title="Stock Price")
    st.plotly_chart(fig_pred)

    # Forecast future prices (next 30 days)
    # We will use the last 30 rows of the training data to predict future prices
    forecast_input = features.tail(30)
    forecast_input_scaled = scaler.transform(forecast_input)

    # Predict future stock prices
    future_pred = model.predict(forecast_input_scaled)

    # Create a date range for the next 30 days
    forecast_dates = pd.date_range(start=stock_data['date'].max() + pd.Timedelta(days=1), periods=30, freq='D')

    # Create a DataFrame for the forecasted values with the correct date range as the index
    forecast_df = pd.DataFrame(future_pred, index=forecast_dates, columns=["Forecast"])

    # Plot forecasted future prices
    st.subheader(f"Forecasted Future Close Prices for {selected_stock}")
    fig_forecast = px.line(title=f"{selected_stock} Forecasted Close Price (Next 30 Days)",
                           x=forecast_df.index, y=forecast_df["Forecast"], 
                           labels={"x": "Date", "y": "Price"})
    fig_forecast.update_layout(xaxis_title="Date", yaxis_title="Stock Price")
    st.plotly_chart(fig_forecast)

    # Show some of the forecasted values
    st.write(f"**Forecasting for the next 30 days**")
    st.write(forecast_df.head())  # Show first few rows of forecast
