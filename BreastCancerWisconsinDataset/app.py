# Task 4 - 27/June/2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier

# Load dataset
data_set = pd.read_csv("BreastCancerWisconsinDataset/BreastCancerDataset.csv")
data_frame = pd.DataFrame(data_set)
print(data_frame.head(10))

# Summary stat of dataset
print(data_set.info())
# print(data_set.describe())

data = data_frame.drop(columns = ['id', 'diagnosis', 'Unnamed: 32'])
target = data_frame['diagnosis']

# Split into numerical and categorical columns
numerical_cols = data.select_dtypes(include = ['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include = ['object', 'category']).columns

# check for NaN details
print(data.isnull().sum()) # No  NaN values

# Scaling the cols
data = StandardScaler().fit_transform(data)
data = pd.DataFrame(data)
print(data)

# Encode the target labels
target = target.map({'M':1, 'B':0})

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# Model Evaluation using Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print(f"\n\nAccuracy score of Prediction = {accuracy_score(y_test, y_predict)*100} %")
print(f"Classification Report \n\n {classification_report(y_test,y_predict)}")

# View Model Prediction vs Actural Data
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
# plt.show()

# Model Evaluation using Decision Trees
model = DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print(f"\n\nAccuracy score of Prediction = {accuracy_score(y_test, y_predict)*100} %")
print(f"Classification Report \n\n {classification_report(y_test,y_predict)}")

# Model Evaluation using Support Vector Classification
model = SVC(kernel = "rbf")
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print(f"\n\nAccuracy score of Prediction = {accuracy_score(y_test, y_predict)*100} %")
print(f"Classification Report \n\n {classification_report(y_test,y_predict)}")

# Model Evaluation using Random Forest
model = RandomForestClassifier(criterion = "entropy", max_depth = 4)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print(f"\n\nAccuracy score of Prediction = {accuracy_score(y_test, y_predict)*100} %")
print(f"Classification Report \n\n {classification_report(y_test,y_predict)}")


# Ensembling basic models
base_learners = [
            ('lr', LogisticRegression(max_iter = 1000)),
            ('svc', SVC(kernel = 'rbf', probability = True))
]

meta_learner = LogisticRegression()

stacked_model = StackingClassifier(
            estimators = base_learners,
            final_estimator = meta_learner,
            cv =5
)

stacked_model.fit(X_train, y_train)
y_predict = stacked_model.predict(X_test)

print(f"\n\nAccuracy score of Prediction = {accuracy_score(y_test, y_predict)*100} %")
print(f"Classification Report \n\n {classification_report(y_test,y_predict)}")
