import pandas as pd

# Load the dataset
df = pd.read_csv("StudentsPerformance.csv")


# Display the first few rows
print(df.head())
print(df.info())  # Check column names and data types

from sklearn.preprocessing import LabelEncoder

# Apply Label Encoding to categorical columns
categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Save the encoders for later use

# View transformed dataset
print(df.head())

from sklearn.model_selection import train_test_split

X = df.drop(columns=['math score'])  # Features
y = df['math score']  # Target variable

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split successfully!")

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Initialize models
decision_tree = DecisionTreeRegressor()
random_forest = RandomForestRegressor(n_estimators=100)

# Train models
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

print("Models trained successfully!")

from sklearn.metrics import mean_absolute_error, r2_score

# Predict values
y_pred_tree = decision_tree.predict(X_test)
y_pred_forest = random_forest.predict(X_test)

# Evaluate accuracy
mae_tree = mean_absolute_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

mae_forest = mean_absolute_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)

print(f"Decision Tree - MAE: {mae_tree}, R² Score: {r2_tree}")
print(f"Random Forest - MAE: {mae_forest}, R² Score: {r2_forest}")

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestRegressor()
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

import matplotlib.pyplot as plt
import numpy as np

feature_importance = random_forest.feature_importances_
features = X_train.columns

sorted_indices = np.argsort(feature_importance)[::-1]
plt.figure(figsize=(8, 4))
plt.bar(features[sorted_indices], feature_importance[sorted_indices], color="green")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Random Forest")
plt.show()

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize Random Forest
rf = RandomForestRegressor()

# Perform Grid Search
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best Parameters:", grid_search.best_params_)

# Retrieve best parameters
best_params = grid_search.best_params_

# Train optimized Random Forest
optimized_rf = RandomForestRegressor(**best_params)
optimized_rf.fit(X_train, y_train)

# Make predictions
y_pred_optimized = optimized_rf.predict(X_test)

# Evaluate new model
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print(f"Optimized Random Forest - MAE: {mae_optimized}, R² Score: {r2_optimized}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.scatter(y_test, y_pred_optimized, color='red', label="Optimized Random Forest Predictions")
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Optimized Random Forest Prediction Accuracy")
plt.legend()
plt.show()

