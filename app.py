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
# for col in num_cols:
#   plt.figure(figsize=(6, 4))
#   sns.boxplot(x=df[col])
#   plt.title(f"Boxplot of {col}")
#   plt.show()

# Removing outliers using IQR
data_df = pd.DataFrame(data, columns=num_cols.tolist() + cat_cols.tolist())

Q1 = data_df[num_cols].quantile(0.25)
Q3 = data_df[num_cols].quantile(0.75)
IQR = Q3 - Q1

filtered_data = data_df[~((data_df[num_cols] < (Q1 - 1.5 * IQR)) | 
                          (data_df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Showing new shape after outlier removal
print("Original shape:", data_df.shape)
print("After outlier removal:", filtered_data.shape)

# modifying our target accordingly
target = target[filtered_data.index]

# Scaling the data
scaler = StandardScaler()
data = scaler.fit_transform(filtered_data)

# Spliting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(data,target, test_size= 0.3, random_state= 42)

# Evaluating with Decision Tree
model = DecisionTreeClassifier(criterion= 'entropy', max_depth= 4)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

acc = accuracy_score(y_test, y_predict)
print(f"\nAccuracy score when testing with Decision Tree = {acc*100} %")

#Evaluating with RandomForest
model = RandomForestClassifier(
    criterion= 'entropy', 
    max_depth=4, 
    n_estimators=200, 
    min_samples_split=4, 
    min_samples_leaf=2,)

model.fit(X_train, y_train)
y_predict = model.predict(X_test)

acc = accuracy_score(y_test, y_predict)
print(f"\nAccuracy score when testing with Random Forest = {acc*100} %") # More generalised

# Evaluating with multi layered Perceptron
model = MLPClassifier(
    activation= 'relu',
    solver= 'adam',
    max_iter= 500,
    hidden_layer_sizes= (10,5),
    random_state= 42,
    verbose=True,           
    n_iter_no_change=10,    
)


model.fit(X_train, y_train)
y_predict = model.predict(X_test)

acc = accuracy_score(y_test, y_predict)
print(f"\nAccuracy score when testing with MLP = {acc*100} %") #Even More generalised