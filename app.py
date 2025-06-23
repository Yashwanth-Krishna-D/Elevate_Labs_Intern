# Task 1 - 23/June/25

# Importing all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# importing models to work with
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Reading the dataset
data_set = pd.read_csv(r"Titanic-Dataset.csv")
df = pd.DataFrame(data_set)
print(df.head())

# Gaining basic details about the dataset
print(df.info()) #df contains both numerical and categorical values
print(df.isnull().sum()) # Shows that df contains NaN values so impute


# Taking only necessary cols for classification
data = df.drop(columns= ["PassengerId", "Survived", "Name", "Ticket"])
target = df["Survived"]

# seperating numerical and categorical cols
num_cols = data.select_dtypes(include = ['int64', 'float64', 'number']).columns

cat_cols = data.select_dtypes(include = ['category', 'object']).columns

# Imputing according to dtype
num_imputer = SimpleImputer(strategy = "mean")
cat_imputer = SimpleImputer(strategy = "most_frequent")

data[num_cols] = num_imputer.fit_transform(data[num_cols])
data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

print(data.isnull().sum()) # After imputing

# Encoding categorical cols
enc = OrdinalEncoder()
data[cat_cols] = enc.fit_transform(data[cat_cols])

# Visualising outliers using box plot
for col in num_cols:
  plt.figure(figsize=(6, 4))
  sns.boxplot(x=df[col])
  plt.title(f"Boxplot of {col}")
  plt.show()