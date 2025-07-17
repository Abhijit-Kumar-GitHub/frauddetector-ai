# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_theme(style='whitegrid')

def plot_class_distribution(df, save_path=None):
    plt.figure(figsize=(8,6))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution: Fraud vs Non-Fraud')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_amount_distribution(df, save_path=None):
    plt.figure(figsize=(8,6))
    sns.histplot(df['Amount'], bins=50, kde=True)
    plt.title('Transaction Amount Distribution')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_time_distribution(df, save_path=None):
    plt.figure(figsize=(8,6))
    sns.histplot(df['Time'], bins=50, kde=True)
    plt.title('Transaction Time Distribution')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_correlation_matrix(df, save_path=None):
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    if save_path:
        plt.savefig(save_path)
    plt.show()
