import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from helper import preprocess_data, feature_engineer, train_test, train_logistic_regression, train_kmeans
from visuals import plot_kmeans_clusters, evaluate_model, visualize_predictions, pie_chart_split

# Streamlit Page Configuration
st.set_page_config(
    page_title="Stock Aunty ko sab maloom hai!",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("styles.css file not found.")

# Sidebar - Upload or fetch data
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("## Aunty ka MUFT ka gyaan pehli baar faideymand!")

option = st.sidebar.radio("Select Data Source:", ["Upload CSV (Kragle)", "Fetch from Yahoo Finance"])
data = None

if option == "Upload CSV (Kragle)":
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type="csv")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Dataset loaded from Kragle 👵")
        except Exception as e:
            st.error(f"File load error: {str(e)}")

elif option == "Fetch from Yahoo Finance":
    ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())
    fetch_data = st.sidebar.button("Fetch Data")

    if fetch_data:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.error("No data returned. Please check the ticker symbol.")
            else:
                st.success(f"Data fetched for {ticker} from Yahoo Finance")
        except Exception as e:
            st.error(f"Data fetch failed: {str(e)}")

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

# App Logic
if data is not None and not data.empty:
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
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.success("Training aur testing split ready hai!")

    elif step == "5. Logistic Regression":
        if "X_train" in st.session_state and "y_train" in st.session_state:
            model_lr = train_logistic_regression(st.session_state.X_train, st.session_state.y_train)
            st.session_state["model_lr"] = model_lr
            st.success("Aunty ne logistic regression train kar diya hai!")
        else:
            st.warning("Pehle Train/Test split karna hoga.")

    elif step == "6. K-Means Clustering":
        kmeans_model = train_kmeans(data)
        plot_kmeans_clusters(data, kmeans_model)
        st.success("K-means se clusters ban gaye!")

    elif step == "7. Evaluation":
        if "model_lr" in st.session_state and "X_test" in st.session_state:
            evaluate_model(st.session_state.model_lr, st.session_state.X_test, st.session_state.y_test)
            st.success("Model ka evaluation ho gaya!")
        else:
            st.warning("Pehle Logistic Regression train karo.")

    elif step == "8. Visualize Results":
        if "model_lr" in st.session_state and "X_test" in st.session_state:
            visualize_predictions(st.session_state.model_lr, st.session_state.X_test, st.session_state.y_test)
            st.success("Results ke graphs taiyaar hain!")
        else:
            st.warning("Pehle Logistic Regression train karo.")
else:
    home()
    st.info("Pehle apna data upload karo ya phir Yahoo se le lo.")
