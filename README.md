
# AI BASED DIABETES PREDICTION SYSTEM

This project is an AI-powered diabetes prediction system that uses machine learning algorithms to analyze medical data and predict the likelihood of an individual developing diabetes. The system aims to provide early risk assessment and personalized preventive measures, allowing individuals to take proactive actions to manage their health.

## Table of Contents
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Prediction](#prediction)
- [Contributors](#contributors)

## Installation
### Required Libraries
Install the necessary libraries as shown below.
```bash
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
import warnings
warnings.simplefilter(action = "ignore")
```
## Uploading the Dataset
Upload the dataset. The dataset used in this project can be found in /content/diabetes.csv. Make sure to replace this path with the correct dataset path in your system.
```bash
df = pd.read_csv("path/to/diabetes.csv")
```
## Data Preprocessing
The dataset undergoes several data preprocessing steps:

- Replacing 0 values with NaN to identify missing values.
- Handling missing values by replacing them with median values.
- Identifying and treating outliers.

## Selecting Relevant Features
New features are created to improve the model's performance:

- NewBMI: Categorizes BMI into different weight categories (e.g., underweight, normal, overweight, etc.).
- NewInsulinScore: Identifies whether the insulin level is normal or abnormal.
- NewGlucose: Categorizes glucose levels as low, normal, high, etc.
## Model Training
A RandomForestClassifier is used to train the model on the preprocessed data.
```bash
from sklearn.ensemble import RandomForestClassifier

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
```
## Model Evaluation
The model's performance is evaluated using various metrics, including accuracy, precision, recall, and the confusion matrix.
```bash
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display other evaluation metrics
print(classification_report(y_test, y_pred))

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
```
## Prediction
The trained model can be used to make predictions on new data points to determine whether an individual is diabetic or not.
```bash
# Example predictions
X_pred1 = [0.6, 0.770186, 0.000, 1.000000, 1.000000, 0.177778, 0.672313, 1.235294, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
y_pred1 = clf.predict([X_pred1])  # Output: 1 (diabetic)

# Perform similar predictions for other data points
```
## Contributors
- [@Bhavadharini-C](https://github.com/Bhavadharini-C)
- [@Gayathri-Chandrasekaran](https://github.com/Gayathri-Chandrasekaran)
- [@Krishnaprabha123](https://github.com/Krishnaprabha123)
- [@Sweda01](https://github.com/Sweda01)
- [@visalivs](https://github.com/visalivs)





