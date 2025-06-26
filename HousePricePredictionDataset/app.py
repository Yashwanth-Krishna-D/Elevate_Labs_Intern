# Task 3 : 26/June/2025

# Importing all necessary preprocessing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#import required models
from sklearn.linear_model import LinearRegression, LogisticRegression

# Loading the dataset
data_set = pd.read_csv("HousePricePredictionDataset/Housing.csv")
data_frame = pd.DataFrame(data_set)

# Checking for summary statistics and description of dataset
print(data_frame.info()) # contains both numerical and categorical entries
print(data_frame.isnull().sum()) # no NaN values are found so no need to impute
print(data_frame.describe()) 

# spliting to data and target
data = data_frame.drop(columns=['price'])
target = data_frame['price']

# Categorical Encoding
numerical_cols = data.select_dtypes(include = ['int64']).columns
categorical_cols = data.select_dtypes(include = ['object', 'category']).columns
enc = OrdinalEncoder()
data[categorical_cols] = enc.fit_transform(data[categorical_cols])

# visulaise the data
# for col in numerical_cols :
#   plt.figure(figsize=(6,4))
#   sns.boxplot(x= data[col], color='skyblue')
#   plt.title(f'Box plot visulization of {col}')
#   plt.xlabel(col)
#   plt.tight_layout()
#   plt.show()

# for col in numerical_cols:
#   plt.figure(figsize=(6,4))
#   sns.histplot(x= data[col], bins=30, kde=True, color='orange')
#   plt.title(f'Box plot visulization of {col}')
#   plt.xlabel(col)
#   plt.ylabel("frequency")
#   plt.tight_layout()
#   plt.show()

# scaling the features
scaler = StandardScaler()
data = scaler.fit_transform(data)
scaled_data = pd.DataFrame(data)
print(scaled_data.head())

# Train test split
X_train,X_test, y_train, y_test = train_test_split(data, target, test_size = 0.3, random_state=42)

# Evaluate Models:

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
predicted_price = model.predict(X_test)

mae = mean_absolute_error(y_test, predicted_price)
mse = mean_squared_error(y_test, predicted_price)
R_sqaure_value = r2_score(y_test, predicted_price)
print(f"The mean absolute error in prediction = {mae}")
print(f"The mean squared error in prediction = {mse}")
print(f"The R square value of prediction = {R_sqaure_value}")

# Visualise prediction accuracy 
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=predicted_price, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()
