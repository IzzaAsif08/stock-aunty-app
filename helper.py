import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        return None

def preprocess_data(df, target_column=None):
    df = df.dropna()
    
    # Separate features and target if classification
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.copy()
        y = None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns.tolist()

def split_data(X, y=None, test_size=0.2, random_state=42):
    if y is not None:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(X, test_size=test_size, random_state=random_state)
