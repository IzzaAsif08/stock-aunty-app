import streamlit as st
import pandas as pd
import yfinance as yf
from helper import preprocess_data, feature_engineer, train_test, train_logistic_regression
#from visuals import plot_kmeans_clusters, evaluate_model, visualize_predictions
import datetime

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
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            data = yf.download(ticker, start=start_date_str, end=end_date_str)
            st.success(f"Data fetched for {ticker} from Yahoo Finance")
        except Exception as e:
            st.error(f"Data fetch failed: {str(e)}")

# Main Interface
st.title("Apki pasandeedah Stock Aunty")

def home():
    st.markdown("## 👋 Khush Amdeed to the *Aunty’s Stock School*!")
    
    # Display welcome GIF
    st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExNXA1dm55emZvYW1jeWFhbGxvZWlwNDdxMzNuejNrdGJ0eWcycXNrNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/P4iv1IkJxwkODZOH8Q/giphy.gif", use_column_width=True)

st.image("download.gif", width=350)
st.markdown("""
    Aunty kehti hain: "Beta, stocks mein sirf luck nahi, logic bhi hota hai. Aao sikhun tumhein!"
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
        st.success("Missing values aur outliers ki safai ho gayi!")

    elif step == "3. Feature Engineering":
        data = feature_engineer(data)
        st.success("Features select aur transform kar diye gaye hain.")

    elif step == "4. Train/Test Split":
        X_train, X_test, y_train, y_test = train_test(data)
        pie_chart_split(X_train, X_test)
        st.success("Training aur testing split ready hai!")

    elif step == "5. Logistic Regression":
        model_lr = train_logistic_regression(X_train, y_train)
        st.success("Aunty ne logistic regression train kar diya hai bas kabhi gharoor nahi kia")

    elif step == "6. K-Means Clustering":
        kmeans_model = train_kmeans(data)
        plot_kmeans_clusters(data, kmeans_model)
        st.success("K-means se clusters ban gaye!")

    elif step == "7. Evaluation":
        evaluate_model(model_lr, X_test, y_test)
        st.success("Model ka evaluation ho gayi")

    elif step == "8. Visualize Results":
        visualize_predictions(model_lr, X_test, y_test)
        st.success("Results ke graphs taiyaar hain")
else:
    home()
    st.info("Pehle apna data upload karo ya phir Yahoo se le lo.")
