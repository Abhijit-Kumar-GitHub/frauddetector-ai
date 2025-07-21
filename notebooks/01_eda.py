# 01_eda.ipynb - Enhanced EDA for Credit Card Fraud Detection

# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.visualization import plot_amount_distribution, plot_amount_distribution_by_class

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# --- Load Dataset ---
DATA_PATH = '../data/creditcard.csv'
df = pd.read_csv(DATA_PATH)

# --- Dataset Overview ---
print("Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nNull Values:\n", df.isnull().sum())
duplicate_count = df.duplicated().sum()
print(f"\nDuplicate Entries: {duplicate_count}")

# Drop duplicates as the similar time values suggest redundant records
df = df.drop_duplicates()
print(f"Data shape after removing duplicates: {df.shape}")

# Separate the data based on fraud or legit transactions
df_fraud = df[df['Class'] == 1]
df_legit = df[df['Class'] == 0]

print("Time Statistics for Fraudulent Transactions:")
print(df_fraud['Time'].describe())

print("\nTime Statistics for Legitimate Transactions:")
print(df_legit['Time'].describe())


# --- Class Distribution ---
class_counts = df['Class'].value_counts()
print("\nClass Distribution:\n", class_counts)
print("\nPercentage Distribution:\n", class_counts / len(df) * 100)
ax = sns.countplot(x='Class', data=df)
ax.set_yscale('log')
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
plt.title('Class Distribution (Log Scale with Annotations)')
total = len(df)
for p in ax.patches:
    count = int(p.get_height())
    percentage = f'{100 * count / total:.5f}%'
    ax.annotate(f'{count}\n({percentage})', (p.get_x() + p.get_width() / 2, p.get_height()),
    ha='center', va='bottom', fontsize=10)
plt.show()

plot_amount_distribution(df)

plot_amount_distribution_by_class(df)

sns.histplot(df['Time'], bins=50, kde=True)
plt.title('Transaction Time Distribution')
plt.show()

sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Transaction Amount by Class')
plt.show()

plt.figure(figsize=(12,6))
sns.kdeplot(data=df, x='Time', hue='Class', common_norm=False, fill=True, palette='husl', alpha=0.5)
plt.title('KDE Plot of Transaction Time by Class')
plt.show()

# --- Correlation Matrix ---
plt.figure(figsize=(12,10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# --- Prepare Data ---
X = df.drop('Class', axis=1)
y = df['Class']

# --- Correlation with Class (Linear) ---
# Useful for logistic reg and other linear models
correlations = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)
print("Top 5 Features by Correlation with Class:\n", correlations.head(5))

# --- Mutual Information (Linear + Non-linear) ---
# Useful for xgboost and other tree-based models
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("Top 5 Features by Mutual Information:\n", mi_series.head(5))

# --- Combined View ---
combined = pd.DataFrame({
    'Correlation': correlations,
    'Mutual Information': mi_series
}).sort_values(by='Mutual Information', ascending=False)

print("Combined Feature Relevance:\n", combined.head(10))

top_features = combined.head(5).index.tolist()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, feature in enumerate(top_features):
    sns.kdeplot(
        data=df, 
        x=feature, 
        hue='Class', 
        fill=True, 
        common_norm=False, 
        palette='Set2', 
        alpha=0.5,
        ax=axes[i]
    )
    axes[i].set_title(f'KDE of {feature} by Class')

# Hide any empty subplots grids if top_features < 6
for j in range(len(top_features), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

print("\nEDA completed. Key visualizations generated and ready for further analysis or saving.")


