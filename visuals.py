import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd

def plot_kmeans_clusters(df, model):
    labels = model.predict(df)
    df = df.copy()
    df['Cluster'] = labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue='Cluster', palette='tab10')
    plt.title("K-Means Clustering")
    st.pyplot(plt.gcf())

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    st.pyplot(plt.gcf())

def visualize_predictions(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_prob, bins=20, color='skyblue')
    plt.title("Predicted Probabilities (Positive Return)")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    st.pyplot(plt.gcf())

def pie_chart_split(X_train, X_test):
    sizes = [len(X_train), len(X_test)]
    labels = ['Train', 'Test']
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    plt.title('Train-Test Split')
    st.pyplot(plt.gcf())
