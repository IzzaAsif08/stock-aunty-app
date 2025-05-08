import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Streamlit Page Configuration
st.set_page_config(
    page_title="Stock Aunty ko sab maloom hai!",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar - Upload or fetch data
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("## Aunty ka MUFT ka gyaan pehli baar faideymand!")

option = st.sidebar.radio("Select Data Source:", ["Upload CSV (Kragle)", "Fetch from Yahoo Finance"])
data = None

if option == "Upload CSV (Kragle)":
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.success("Dataset loaded from Kragle 👵")

elif option == "Fetch from Yahoo Finance":
    ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")
    start_date = st.sidebar.date_input("Start Date", min_value=datetime.date(2000, 1, 1))
    end_date = st.sidebar.date_input("End Date", max_value=datetime.date.today())

    if st.sidebar.button("Fetch Data"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            st.success(f"Data fetched for {ticker} from Yahoo Finance")
        except Exception as e:
            st.error(f"Data fetch failed: {str(e)}")

# Helper Functions
def preprocess_data(data):
    data.dropna(inplace=True)
    return data

def feature_engineer(data):
    data['Date'] = pd.to_datetime(data.index)
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    return data

def train_test(data):
    X = data[['Year', 'Month']]
    y = data['Close']  # Predicting the 'Close' column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_kmeans(data):
    X = data[['Year', 'Month']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    return kmeans

# Visuals Functions
def plot_kmeans_clusters(data, kmeans_model):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Year'], data['Month'], c=kmeans_model.labels_, cmap='viridis')
    plt.title("K-Means Clusters")
    plt.xlabel("Year")
    plt.ylabel("Month")
    st.pyplot(plt)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.text(classification_report(y_test, y_pred))

def visualize_predictions(model, X_test, y_test):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label="Actual", color='blue')
    plt.plot(y_test.index, y_pred, label="Predicted", color='red')
    plt.legend()
    plt.title("Actual vs Predicted Stock Prices")
    st.pyplot(plt)

def pie_chart_split(X_train, X_test):
    labels = ['Train', 'Test']
    sizes = [len(X_train), len(X_test)]
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ffcc99'])
    plt.title("Train/Test Split")
    st.pyplot(plt)

# Main Interface
def home():
    st.markdown("## 👋 Khush Amdeed to the *Aunty’s Stock School*!")
    st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExNXA1dm55emZvYW1jeWFhbGxvZWlwNDdxMzNuejNrdGJ0eWcycXNrNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/P4iv1IkJxwkODZOH8Q/giphy.gif", width=200)

st.image("aunty stock.png", width=100)
st.markdown("""
    Aksar jo desi aunties hoti hain, unka gyaan sunna thora frustrating ho sakta hai.
    Wo hamesha koi na koi taana deti hain, ya phir bas zyada hi poking kar leti hain.
    Lekin yeh Aunty alag hai! Inki poori koshish hai aapki madad karna, aur stock market ko samajhne mein apko raah dikhana.
    Yeh application aapko stock data ko analyse karne, seekhne aur invest karne mein madad karegi. 
    Toh aaiye, is Aunty ke saath apna stock journey shuru karein – pehli baar bilkul friendly tareeke se!
""")

if data is not None:
    step = st.radio("Select ML Step", [
        "1. Preview Data",
        "2. Preprocess",
        "3. Feature Engineering",
        "4. Train/Test Split",
        "5. Logistic Regression",
        "6. K-Means Clustering",
        "7. Evaluation",
        "8. Visualize Results"
    ])

    if step == "1. Preview Data":
        st.write(data.head())
        st.success("Data dekh liya, shabash!")

    elif step == "2. Preprocess":
        data = preprocess_data(data)
        st.write(data.head())
        st.success("Missing values aur outliers ki safai ho gayi!")

    elif step == "3. Feature Engineering":
        data = feature_engineer(data)
        st.write(data.head())
        st.success("Features select aur transform kar diye gaye hain.")

    elif step == "4. Train/Test Split":
        X_train, X_test, y_train, y_test = train_test(data)
        pie_chart_split(X_train, X_test)
        st.session_state.update({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        })
        st.success("Training aur testing split ready hai!")

    elif step == "5. Logistic Regression":
        if "X_train" in st.session_state:
            model_lr = train_logistic_regression(st.session_state.X_train, st.session_state.y_train)
            st.session_state.model_lr = model_lr
            st.success("Aunty ne logistic regression train kar diya hai!")
        else:
            st.warning("Pehle Train/Test split karna hoga.")

    elif step == "6. K-Means Clustering":
        kmeans_model = train_kmeans(data)
        plot_kmeans_clusters(data, kmeans_model)
        st.success("K-means se clusters ban gaye!")

    elif step == "7. Evaluation":
        if "model_lr" in st.session_state:
            evaluate_model(st.session_state.model_lr, st.session_state.X_test, st.session_state.y_test)
            st.success("Model ka evaluation ho gaya!")
        else:
            st.warning("Pehle Logistic Regression train karo.")

    elif step == "8. Visualize Results":
        if "model_lr" in st.session_state:
            visualize_predictions(st.session_state.model_lr, st.session_state.X_test, st.session_state.y_test)
            st.success("Results ke graphs taiyaar hain!")
        else:
            st.warning("Pehle Logistic Regression train karo.")
else:
    home()
    st.info("Pehle apna data upload karo ya phir Yahoo se le lo.")
