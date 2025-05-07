import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

from helper import load_data, preprocess_data, split_data
from visuals import pie_chart_split, plot_kmeans_clusters, evaluate_model, visualize_predictions

st.set_page_config(page_title="ML Visualizer", layout="wide")
st.title("🧠 Machine Learning Visualizer")

# Sidebar
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        st.write("### Raw Data Preview")
        st.dataframe(df)

        task_type = st.sidebar.selectbox("Select Task Type", ["Classification", "Clustering"])
        target_col = None

        if task_type == "Classification":
            target_col = st.sidebar.selectbox("Select Target Column", df.columns)

        # Preprocess
        X, y, feature_names = preprocess_data(df, target_col)
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Pie chart of split
        pie_chart_split(X_train, X_test)

        if task_type == "Classification":
            model = LogisticRegression()
            model.fit(X_train, y_train)
            st.success("✅ Logistic Regression model trained!")

            evaluate_model(model, X_test, y_test)
            visualize_predictions(model, X_test, y_test)

        elif task_type == "Clustering":
            num_clusters = st.sidebar.slider("Select Number of Clusters (k)", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(X)

            plot_kmeans_clusters(pd.DataFrame(X, columns=feature_names), kmeans)

    else:
        st.error("❌ Failed to load CSV. Please check the file format.")
else:
    st.info("👈 Upload a CSV file to get started.")
