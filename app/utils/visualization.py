"""
Author: Onur Sasmaz
Copyright (c) 2024 Onur Sasmaz. All Rights Reserved.

"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from sklearn.model_selection import train_test_split
import numpy as np

def plot_data_distribution(df):
    """
    Plots the distribution of sentiment labels in a DataFrame.

    Args:
    df (DataFrame): Input DataFrame containing 'sentiment' column.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig('data_distribution.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plots a confusion matrix based on true and predicted labels.

    Args:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    labels (array-like): List of unique labels/classes.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

def show_classification_report(y_true, y_pred, labels):
    """
    Shows the classification report as a DataFrame and saves it to a CSV file.

    Args:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    labels (array-like): List of unique labels/classes.
    """
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report)
    df_report.to_csv('classification_report.csv', index=True)

def visualize_model_performance(df, model_path, encoder_path):
    """
    Visualizes the performance of a machine learning model on sentiment analysis.

    Args:
    df (DataFrame): Input DataFrame with 'review' and 'sentiment' columns.
    model_path (str): File path to the trained model (joblib format).
    encoder_path (str): File path to the label encoder (joblib format).
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)
    
    # Load model and encoder
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Encode y_test to numerical values
    y_true = label_encoder.transform(y_test)
    
    # Ensure y_pred contains only known labels
    print("Unique values in y_pred before decoding:", np.unique(y_pred))
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    print("Unique values in y_pred_decoded:", np.unique(y_pred_decoded))
    print("Label encoder classes:", label_encoder.classes_)
    y_pred_encoded = label_encoder.transform(y_pred_decoded)
    
    # Labels as integers
    labels = label_encoder.transform(label_encoder.classes_)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred_encoded, labels)
    
    # Show classification report
    show_classification_report(y_true, y_pred_encoded, label_encoder.classes_)
