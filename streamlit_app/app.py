# streamlit_app/app.py
import streamlit as st
import pandas as pd
from components import (
    plot_class_distribution,
    plot_amount_distribution,
    plot_amount_distribution_by_class,
    plot_log_amount_distribution_by_class,
    plot_fraud_frequency_per_amount_bin,
    plot_fraud_rate_vs_total_by_amount_bin,
    plot_time_distribution,
    plot_time_kde_by_class,
    plot_correlation_heatmap,
    plot_filtered_correlation_heatmap,
    plot_top_features_kde
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
    st.write("Null values:", df.isnull().sum().sum())

    st.header("Distributions & Correlations")
    plot_class_distribution(df)
    plot_amount_distribution(df)
    # plot_fraud_frequency_per_amount_bin(df)
    plot_fraud_rate_vs_total_by_amount_bin(df)
    # plot_log_amount_distribution_by_class(df)
    # plot_amount_distribution_by_class(df)
    plot_time_distribution(df)
    plot_time_kde_by_class(df)
    # plot_correlation_heatmap(df)
    plot_filtered_correlation_heatmap(df)
    st.header("Top Feature Distributions by Class")
    top_features = ['V17', 'V14', 'V12', 'V10', 'V11']
    plot_top_features_kde(df, top_features)

if __name__ == "__main__":
    main()
