# Task 1: Load and Explore the Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully.\n")
except Exception as e:
    print(f"Error loading dataset: {e}")


print("First 5 rows of the dataset:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


df.dropna(inplace=True)

# Task 2


print("\nBasic statistics:")
print(df.describe())


grouped = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(grouped)


print("\nObservations:")
print("→ Setosa has the smallest petal measurements.")
print("→ Virginica tends to have the largest measurements overall.")
print("→ Sepal length and petal length seem to vary significantly across species.")

# Task 3: Data Visualization

sns.set(style="whitegrid")


df['index'] = range(len(df))
df_sorted = df.sort_values(by='index')
plt.figure(figsize=(10, 5))
plt.plot(df_sorted['index'], df_sorted['petal length (cm)'], label='Petal Length')
plt.title('Simulated Time-Series of Petal Length')
plt.xlabel('Index')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()


plt.figure(figsize=(8, 5))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'], palette='viridis')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

*
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='deep')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()