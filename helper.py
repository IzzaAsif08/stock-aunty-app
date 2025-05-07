import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.dropna()
    df = df.select_dtypes(include=['number'])
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def feature_engineer(df):
    if 'Close' in df.columns:
        df['Return'] = df['Close'].pct_change()
        df.dropna(inplace=True)
    return df

def train_test(df):
    if 'Return' not in df.columns:
        raise ValueError("Data must contain a 'Return' column for target variable.")
    X = df.drop('Return', axis=1)
    y = (df['Return'] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_kmeans(df, n_clusters=3):
    if df.empty or len(df.columns) == 0:
        raise ValueError("Dataframe is empty or lacks columns for clustering.")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_scaled)
    return kmeans
