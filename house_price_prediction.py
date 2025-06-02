import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load dataset
data_path = r"C:\Users\polas\OneDrive\Desktop\INTERNSHIP(Next24tech)\TASK 1\housing.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError("Dataset not found. Please place 'housing.csv' in the 'data/' folder.")

# Read data
df = pd.read_csv(data_path)
print("First 5 rows of the dataset:")
print(df.head())

#To rename the column
df = df.rename(columns={"median_house_value": "price"})

# Drop missing values
df = df.dropna()

# Visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop("price", axis=1)
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R² Score: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")

# Save model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "house_price_model.pkl"))
print("Model saved to models/house_price_model.pkl")
