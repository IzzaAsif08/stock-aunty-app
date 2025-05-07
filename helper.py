import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date")
    
    # Remove missing values
    df = df.dropna()

    # Optional: filter to one stock if your dataset has multiple tickers
    if "Ticker" in df.columns:
        df = df[df["Ticker"] == "AAPL"]  # You can change this

    # Feature engineering
    df["Return"] = df["Close"].pct_change()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["Volatility"] = df["Close"].rolling(window=10).std()

    df = df.dropna()

    # Features and target
    X = df[["Open", "High", "Low", "Volume", "MA_10", "Volatility"]]
    y = df["Close"]

    return X, y, df

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    return model, mse, preds, y_test
