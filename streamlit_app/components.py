# streamlit_app/components.py
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style='whitegrid')

def plot_class_distribution(df: pd.DataFrame):
    class_counts = df['Class'].value_counts()
    print("\nClass Distribution:\n", class_counts)
    print("\nPercentage Distribution:\n", class_counts / len(df) * 100)
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax)
    ax.set_yscale('log')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.set_title('Class Distribution (Log Scale with Annotations)')
    total = len(df)
    for p in ax.patches:
        count = int(p.get_height())
        percentage = f'{100 * count / total:.5f}%'
        ax.annotate(f'{count}\n({percentage})', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10)
    st.pyplot(fig)

def plot_fraud_rate_vs_total_by_amount_bin(df, bins=[0, 10, 50, 100, 500, 1000, 5000, 10000, 25000]):
    df['AmountBin'] = pd.cut(df['Amount'], bins=bins)

    bin_counts = df['AmountBin'].value_counts().sort_index()
    fraud_counts = df[df['Class'] == 1]['AmountBin'].value_counts().sort_index()
    
    fraud_rate = (fraud_counts / bin_counts).fillna(0) * 100

    fig, ax1 = plt.subplots(figsize=(14,8))

    ax1.bar(bin_counts.index.astype(str), bin_counts.values, color='teal', alpha=0.7)
    ax1.set_ylabel('Total Transaction Count (log scale)')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(bin_counts.index.astype(str), fraud_rate.values, color='red', marker='o', linewidth=2, label='Fraud Rate (%)')
    ax2.set_ylabel('Fraud Rate (%)')
    
    ax1.set_title('Transaction Count and Fraud Rate per Amount Bin')
    ax1.set_xlabel('Transaction Amount Bins')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    st.pyplot(fig)


def plot_fraud_frequency_per_amount_bin(df, bins=[0, 10, 50, 100, 500, 1000, 5000, 10000, 25000]):
    df['AmountBin'] = pd.cut(df['Amount'], bins=bins)
    
    fraud_rate = df.groupby('AmountBin')['Class'].mean() * 100  # Percentage of frauds
    count_per_bin = df['AmountBin'].value_counts().sort_index()

    fig, ax1 = plt.subplots(figsize=(12,6))

    ax2 = ax1.twinx()
    fraud_rate.plot(kind='line', marker='o', color='red', ax=ax1, label='Fraud Rate (%)')
    count_per_bin.plot(kind='bar', alpha=0.5, ax=ax2, color='blue', label='Total Transactions')

    ax1.set_xlabel('Transaction Amount Bins')
    ax1.set_ylabel('Fraud Rate (%)', color='red')
    ax2.set_ylabel('Total Transactions', color='skyblue')
    ax1.set_title('Fraud Frequency and Total Transactions per Amount Bin')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    st.pyplot(fig)

def plot_amount_distribution(df):
    log_amount = np.log1p(df['Amount'])
    fig, ax = plt.subplots(figsize=(12,6))
    sns.histplot(log_amount, bins=50, kde=True, ax=ax)
    ax.set_title('Transaction Amount Distribution (Log Scale)')
    ticks = ax.get_xticks()
    ax.set_xticklabels([f'{np.expm1(t):.0f}' for t in ticks])
    ax.set_xlabel('Transaction Amount')
    ax.set_ylabel('Count')
    fig.tight_layout()
    st.pyplot(fig)

def plot_amount_distribution_by_class(df):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.kdeplot(data=df, x='Amount', hue='Class', fill=True, common_norm=False, ax=ax, palette='Set2')
    ax.set_title('Transaction Amount Distribution by Class')
    st.pyplot(fig)

def plot_log_amount_distribution_by_class(df: pd.DataFrame):
    df = df.copy()
    df['LogAmount'] = np.log1p(df['Amount'])

    fig, ax = plt.subplots(figsize=(12,6))
    sns.kdeplot(data=df, x='LogAmount', hue='Class', fill=True, common_norm=False, ax=ax, palette='Set2')
    ax.set_title('Log-Scaled Transaction Amount Distribution by Class')
    ax.set_xlabel('Transaction Amount')

    ticks = ax.get_xticks()
    ax.set_xticklabels([f'{np.expm1(t):.0f}' for t in ticks])

    st.pyplot(fig)

def plot_time_distribution(df):
    fig, ax = plt.subplots()
    sns.histplot(df['Time'], bins=50, kde=True, ax=ax)
    ax.set_title('Transaction Time Distribution')
    st.pyplot(fig)

def plot_time_kde_by_class(df):
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x='Time', hue='Class', fill=True, common_norm=False, ax=ax, palette='husl')
    ax.set_title('KDE Plot of Transaction Time by Class')
    st.pyplot(fig)

def plot_correlation_heatmap(df: pd.DataFrame):
    st.subheader("Correlation Heatmap")
    # Filter only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', linewidths=0.5, ax=ax, annot=True, fmt=".2f")
    ax.set_title('Correlation Matrix Heatmap (Numeric Features Only)')
    st.pyplot(fig)

def plot_filtered_correlation_heatmap(df: pd.DataFrame, threshold: float = 0.1):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st

    st.subheader("Correlation Heatmap (Filtered by Correlation with Class)")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    # Filter columns where correlation with Class is above threshold
    corr_target = corr['Class'].abs()
    relevant_features = corr_target[corr_target > threshold].index
    filtered_corr = corr.loc[relevant_features, relevant_features]

    if len(filtered_corr.columns) <= 1:
        st.write("No features have correlation above the threshold with Class.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(filtered_corr, cmap='coolwarm', linewidths=0.5, annot=True, fmt=".2f", ax=ax)
    ax.set_title(f'Correlation Heatmap (Features with |Correlation| > {threshold})')
    st.pyplot(fig)


def plot_top_features_kde(df, top_features):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,12))
    axs = axs.flatten()

    for idx, feature in enumerate(top_features):
        sns.kdeplot(data=df, x=feature, hue='Class', fill=True, common_norm=False, ax=axs[idx], palette='Set2', alpha=0.5)
        axs[idx].set_title(f'KDE of {feature} by Class')

    for j in range(len(top_features), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    st.pyplot(fig)
