import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
# Load the dataset
try:
    df = pd.read_csv('stocks.csv')
except FileNotFoundError:
    print("Error: 'stocks.csv' not found. Please make sure the file is in the same directory.")
    exit()

# Display the head and some info
print("Data Head:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nData Description:")
print(df.describe())


# Check for missing values
print("\nMissing values before handling:")
print(df.isnull().sum())
df.dropna(inplace = True)
print("\nMissing values after handling:")
print(df.isnull().sum())

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])
# Setting the Date column as index
df = df.set_index('Date')
print("\nData types after handling:")
print(df.dtypes)

#check for duplicate rows
print(f"\nNumber of duplicate rows is {df.duplicated().sum()}")
#remove duplicate rows
df.drop_duplicates(inplace = True)

# Feature Engineering
df['SMA_7'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=7).mean())
df['SMA_21'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=21).mean())
df['Price_Change'] = df.groupby('Ticker')['Close'].diff()
df['Volatility'] = df.groupby('Ticker')['Price_Change'].transform(lambda x: x.rolling(window=7).std())


# Display first few rows with new features
print("\nData with Engineered Features:")
print(df.head())
df.dropna(inplace = True)

# Visualization
tickers = df['Ticker'].unique()
for ticker in tickers:
    df_ticker = df[df['Ticker'] == ticker]
    plt.figure(figsize=(15, 8))
    plt.plot(df_ticker['Close'], label='Close Price')
    plt.plot(df_ticker['SMA_7'], label='7-Day SMA')
    plt.plot(df_ticker['SMA_21'], label='21-Day SMA')
    plt.title(f'Stock Close Price and Moving Averages for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot for Volatility
    plt.figure(figsize=(15, 6))
    plt.plot(df_ticker['Volatility'], label='Volatility')
    plt.title(f'Stock Volatility for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Distribution plot for volume
    plt.figure(figsize=(15,6))
    sns.distplot(df_ticker['Volume'])
    plt.title(f'Distribution of Volume for {ticker}')
    plt.show()


# Correlation matrix visualization - For each Ticker
for ticker in tickers:
    df_ticker = df[df['Ticker'] == ticker]
    df_corr = df_ticker.drop('Ticker', axis = 1)
    plt.figure(figsize=(12,8))
    correlation_matrix = df_corr.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f"Correlation Matrix of Stock Data for {ticker}")
    plt.show()

# Prepare Data for Machine Learning
X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_7', 'SMA_21', 'Price_Change', 'Volatility', 'Ticker']]  # Features
y = df['Close']  # Target

# Split Data by Ticker
X_train_dict = {}
X_test_dict = {}
y_train_dict = {}
y_test_dict = {}

for ticker in tickers:
    df_ticker = df[df['Ticker'] == ticker]
    X_ticker = df_ticker[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_7', 'SMA_21', 'Price_Change', 'Volatility']]
    y_ticker = df_ticker['Close']
    X_train, X_test, y_train, y_test = train_test_split(X_ticker, y_ticker, test_size=0.2, random_state=42, shuffle=False)
    X_train_dict[ticker] = X_train
    X_test_dict[ticker] = X_test
    y_train_dict[ticker] = y_train
    y_test_dict[ticker] = y_test

# Scale data and build model for each ticker
model_lr_dict = {}
y_pred_lr_dict = {}
mse_lr_dict = {}
r2_lr_dict = {}

model_rf_dict = {}
y_pred_rf_dict = {}
mse_rf_dict = {}
r2_rf_dict = {}


for ticker in tickers:
    #Scale Data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_dict[ticker])
    X_test_scaled = scaler.transform(X_test_dict[ticker])

    #Linear Regression
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train_dict[ticker])
    y_pred_lr = model_lr.predict(X_test_scaled)
    mse_lr = mean_squared_error(y_test_dict[ticker], y_pred_lr)
    r2_lr = r2_score(y_test_dict[ticker], y_pred_lr)

    model_lr_dict[ticker] = model_lr
    y_pred_lr_dict[ticker] = y_pred_lr
    mse_lr_dict[ticker] = mse_lr
    r2_lr_dict[ticker] = r2_lr

    print(f"\nLinear Regression Results for {ticker}:")
    print(f"Mean Squared Error: {mse_lr}")
    print(f"R-squared: {r2_lr}")

    #Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train_dict[ticker])
    y_pred_rf = model_rf.predict(X_test_scaled)
    mse_rf = mean_squared_error(y_test_dict[ticker], y_pred_rf)
    r2_rf = r2_score(y_test_dict[ticker], y_pred_rf)

    model_rf_dict[ticker] = model_rf
    y_pred_rf_dict[ticker] = y_pred_rf
    mse_rf_dict[ticker] = mse_rf
    r2_rf_dict[ticker] = r2_rf

    print(f"\nRandom Forest Results for {ticker}:")
    print(f"Mean Squared Error: {mse_rf}")
    print(f"R-squared: {r2_rf}")

# Plotting predictions vs actuals for each Ticker

    plt.figure(figsize=(15, 6))
    plt.plot(y_test_dict[ticker].index, y_test_dict[ticker].values, label='Actual Close Prices', color = 'blue')
    plt.plot(y_test_dict[ticker].index, y_pred_lr_dict[ticker], label='Predicted Close Prices (Linear Regression)', color = 'green', linestyle = 'dashed')
    plt.title(f'Actual vs Predicted Close Prices (Linear Regression) for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting predictions vs actuals - Random Forest
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_dict[ticker].index, y_test_dict[ticker].values, label='Actual Close Prices', color = 'blue')
    plt.plot(y_test_dict[ticker].index, y_pred_rf_dict[ticker], label='Predicted Close Prices (Random Forest)', color = 'red', linestyle = 'dashed')
    plt.title(f'Actual vs Predicted Close Prices (Random Forest) for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()