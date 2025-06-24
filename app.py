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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Utility functions
def show_df(data_frame):
    return data_frame.head()

def show_info(data_frame):
    return data_frame.info()

def show_summary(data_frame):
    return data_frame.describe()

def show_shape(data_frame):
    return data_frame.shape

def show_missing_entries(data_frame):
    return data_frame.isnull().sum()

def numerical_columns(data):
    return data.select_dtypes(include=['int64', 'float64', 'number']).columns

def categorical_columns(data):
    return data.select_dtypes(include=['category', 'object']).columns

def imputer(data):
    num_cols = numerical_columns(data)
    cat_cols = categorical_columns(data)

    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    data[num_cols] = num_imputer.fit_transform(data[num_cols])
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])
    return data

def encoder(data):
    cat_cols = categorical_columns(data)
    data[cat_cols] = OrdinalEncoder().fit_transform(data[cat_cols])
    return data

def box_plot(data):
    for col in numerical_columns(data):
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=data[col])
        plt.title(f"Boxplot of {col}")
        plt.show()

def remove_outliers(data):
    num_cols = numerical_columns(data)
    Q1 = data[num_cols].quantile(0.25)
    Q3 = data[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    filtered_data = data[~((data[num_cols] < (Q1 - 1.5 * IQR)) | (data[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return filtered_data

def scale_data(data):
    return StandardScaler().fit_transform(data)

# Main pipeline
df = pd.read_csv("Titanic-Dataset.csv")
target = df["Survived"]
data = df.drop(columns=["PassengerId", "Survived", "Name", "Ticket"])

# Imputation
data = imputer(data)

# Encoding
data = encoder(data)

# Outlier Removal
data = remove_outliers(data)
target = target[data.index]

# Scaling
data = scale_data(data)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# Model Evaluation
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy score with {model_name} = {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))

# Decision Tree
evaluate_model(DecisionTreeClassifier(criterion='entropy', max_depth=4), "Decision Tree")

# Random Forest
evaluate_model(RandomForestClassifier(
    criterion='entropy', max_depth=4, n_estimators=200,
    min_samples_split=4, min_samples_leaf=2
), "Random Forest")

# MLP Classifier
evaluate_model(MLPClassifier(
    activation='relu', solver='adam', max_iter=500,
    hidden_layer_sizes=(10, 5), random_state=42,
    verbose=True, n_iter_no_change=10
), "MLP Classifier")
