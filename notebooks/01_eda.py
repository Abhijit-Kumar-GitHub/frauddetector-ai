# 01_eda.ipynb - Enhanced EDA for Credit Card Fraud Detection

# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.manifold import TSNE

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# --- Load Dataset ---
DATA_PATH = '../data/creditcard.csv'
df = pd.read_csv(DATA_PATH)

# --- Dataset Overview ---
print("Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nNull Values:\n", df.isnull().sum())
print("\nDuplicate Entries:", df.duplicated().sum())

# --- Class Distribution ---
class_counts = df['Class'].value_counts()
print("\nClass Distribution:\n", class_counts)
print("\nPercentage Distribution:\n", class_counts / len(df) * 100)

sns.countplot(x='Class', data=df)
plt.title('Class Distribution: Fraud vs Non-Fraud')
plt.show()

# --- Univariate Analysis ---
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Transaction Amount Distribution')
plt.show()

sns.histplot(df['Time'], bins=50, kde=True)
plt.title('Transaction Time Distribution')
plt.show()

# --- Class-wise Amount Distribution ---
sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Transaction Amount by Class')
plt.show()

# --- Time vs Class Visualization ---
plt.figure(figsize=(12,6))
sns.histplot(data=df, x='Time', hue='Class', bins=50, kde=True, palette='husl', element='step')
plt.title('Time Distribution by Class')
plt.show()

# --- Correlation Matrix ---
plt.figure(figsize=(12,10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# --- Boxplots for V1-V28 grouped by Class ---
v_features = [f'V{i}' for i in range(1, 29)]

for feature in v_features:
    plt.figure(figsize=(8,4))
    sns.boxplot(x='Class', y=feature, data=df)
    plt.title(f'{feature} by Class')
    plt.show()

# --- Optional: t-SNE Visualization of V1-V28 ---
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(df[v_features])

tsne_df = pd.DataFrame(np.column_stack((X_tsne, df['Class'])), columns=['Dim1', 'Dim2', 'Class'])
plt.figure(figsize=(10,6))
sns.scatterplot(x='Dim1', y='Dim2', hue='Class', data=tsne_df, palette='Set1', alpha=0.7)
plt.title('t-SNE Projection of PCA Features')
plt.show()

print("\nEDA completed. Key visualizations generated and ready for further analysis or saving.")


