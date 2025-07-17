# streamlit_app/components.py

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style='whitegrid')

def plot_class_distribution(df: pd.DataFrame):
    st.subheader("Class Distribution: Fraud vs Non-Fraud")
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax)
    st.pyplot(fig)

def plot_amount_distribution(df: pd.DataFrame):
    st.subheader("Transaction Amount Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Amount'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

def plot_time_distribution(df: pd.DataFrame):
    st.subheader("Transaction Time Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Time'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

def plot_correlation_heatmap(df: pd.DataFrame):
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(df.corr(), cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)
