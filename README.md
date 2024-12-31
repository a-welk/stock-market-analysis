
# Stock Market Data Analysis and Prediction

## Description:
This Python-based project is designed to perform a comprehensive analysis of stock market data, providing valuable insights through visualization and predictive modeling. It reads stock data from a CSV file, preprocesses it, performs feature engineering, visualizes key trends, and builds predictive models for stock prices. The project is designed to be a flexible platform for exploring and modeling time series data for different stocks, allowing for visual comparisons.

**Purpose:**

The primary goals of this project are:

-   **Data Exploration:** Understand the behavior of stock prices over time, identify trends, and analyze volatility.
    
-   **Visualization:** Represent stock data visually to aid understanding and identify patterns.
    
-   **Feature Engineering:** Create new features to improve analysis and prediction capabilities.
    
-   **Predictive Modeling:** Build simple machine learning models to predict future stock prices.
    
-   **Comparison Across Stocks:** Analyze and compare multiple stocks using combined visualizations, enabling direct comparisons.
    

**Key Functionality:**

1.  **Data Loading:** Reads stock data from a CSV file (stock_data.csv). The file is expected to contain columns for 'Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', and 'Volume'.
    
2.  **Data Cleaning:**
    
    -   Handles missing values by dropping rows containing NaN.
        
    -   Removes duplicate rows.
        
    -   Converts the 'Date' column to datetime objects and sets it as the index of the DataFrame.
        
3.  **Feature Engineering:**
    
    -   Calculates Simple Moving Averages (SMA) over 7 and 21 days (SMA_7, SMA_21).
        
    -   Computes daily price changes (Price_Change).
        
    -   Calculates rolling volatility (standard deviation of price changes over 7 days).
        
    -   The above feature engineering is done on each stock independently, rather than across all stocks.
        
4.  **Data Visualization:**
    
    -   **Stock Close Prices:** Time series plot of closing prices for all stocks on the same graph.
        
    -   **Moving Averages:** Time series plots of the 7-day and 21-day SMA for all stocks on the same graph.
        
    -   **Volatility:** Time series plot of volatility for all stocks on the same graph.
        
    -   **Volume Distribution:** Distribution plots (histograms) of trading volumes for all stocks on the same graph.
        
    -   **Correlation Matrix:** Heatmap of the correlation matrix of features calculated for each stock independently.
        
5.  **Predictive Modeling:**
    
    -   **Data Splitting:** Splits the dataset into training and test sets, with time series shuffle set to false to prevent look ahead bias.
        
    -   **Data Scaling:** Scales the data using MinMaxScaler
        
    -   **Linear Regression:** Builds and evaluates a Linear Regression model to predict close prices for each stock independently.
        
    -   **Random Forest:** Builds and evaluates a Random Forest Regressor model to predict close prices for each stock independently.
        
    -   **Prediction Visualization:** Plots actual vs. predicted closing prices for the test set for each model (both linear regression and random forest), with each stock represented on the same plot to provide a visual way to compare actual and predicted prices.
        
6.  **Combined Visualizations:**
    
    -   All time-series plots show the data for all stocks on the same graph, distinguished by different line styles and colors, making it easy to compare stocks.
  
# Disclaimer: This project is for practice and educational purpose only. The predictive models should not be used for actual invenstment decisions.
