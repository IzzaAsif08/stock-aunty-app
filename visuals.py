import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def pie_chart_split(X_train, X_test):
    """
    Displays a pie chart showing the distribution of training and testing datasets.
    """
    labels = ['Train', 'Test']
    sizes = [len(X_train), len(X_test)]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=['#2B3A67', '#E84855'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Train-Test Split')
    st.pyplot(fig)
    plt.clf()  # Clear the figure to avoid overlapping plots

def plot_kmeans_clusters(df, kmeans_model):
    """
    Plots the clusters found by the KMeans model.
    """
    df = df.copy()
    if hasattr(kmeans_model, 'labels_'):
        df['Cluster'] = kmeans_model.labels_
    else:
        st.error("KMeans model has not been fitted yet.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue='Cluster', palette='Set1', ax=ax)
    ax.set_title('K-Means Clustering')
    st.pyplot(fig)
    plt.clf()

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model with a confusion matrix and classification report.
    """
    try:
        predictions = model.predict(X_test)
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.clf()

    report = classification_report(y_test, predictions, output_dict=True)
    st.write("### Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

def visualize_predictions(model, X_test, y_test):
    """
    Visualizes the prediction probabilities of the model.
    """
    try:
        y_pred = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        st.error(f"Error getting prediction probabilities: {e}")
        return

    df_viz = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted Probability': y_pred
    })

    st.write("### Prediction Probabilities")
    st.line_chart(df_viz)
