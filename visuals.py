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
    ax.pie(sizes, labels=labels, colors=['#2B3A67', '#E84855'], autopct='%1.1f%%')
    ax.set_title('Train-Test Split')
    st.pyplot(fig)

def plot_kmeans_clusters(df, kmeans_model):
    """
    Plots the clusters found by the KMeans model.
    """
    df = df.copy()
    df['Cluster'] = kmeans_model.labels_
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue='Cluster', palette='Set1')
    plt.title('K-Means Clustering')
    st.pyplot(plt.gcf())

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model with a confusion matrix and classification report.
    """
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    
    st.write("### Confusion Matrix") 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
    st.pyplot(plt.gcf())
    
    st.write("### Classification Report") 
    report = classification_report(y_test, predictions, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

def visualize_predictions(model, X_test, y_test):
    """
    Visualizes the predictions from the model.
    """
    y_pred = model.predict_proba(X_test)[:, 1]
    df_viz = pd.DataFrame({'Actual': y_test.values, 'Predicted Probability': y_pred})
    st.line_chart(df_viz)
