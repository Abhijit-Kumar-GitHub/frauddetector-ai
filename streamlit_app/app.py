# streamlit_app/app.py

import streamlit as st
import pandas as pd
from components import (
    plot_class_distribution,
    plot_amount_distribution,
    plot_time_distribution,
    plot_correlation_heatmap
)

@st.cache_data
def load_data():
    return pd.read_csv('../data/creditcard.csv')

def main():
    st.title("💳 Credit Card Fraud Detection - EDA Dashboard")
    df = load_data()

    st.header("Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Null values:\n", df.isnull().sum())

    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_time_distribution(df)
    plot_correlation_heatmap(df)

if __name__ == "__main__":
    main()

