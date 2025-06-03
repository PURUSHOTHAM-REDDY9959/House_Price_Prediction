# House_Price_Prediction

# 🏠 House Price Prediction System

## DESCRIPTION :

This project presents a **House Price Prediction System** using **Linear Regression**, built with Python and scikit-learn. The goal is to predict house prices based on a variety of features such as median income, housing median age, number of rooms, location, and more. The model is trained on a cleaned and preprocessed version of the **California housing dataset**.

This project serves as a practical example of how machine learning models are trained, evaluated, and saved for reuse. It also includes data visualization to understand feature correlations and improve model insights.he aim of the project is to create a  model that can predict house prices based on various features present in the dataset, such as location, income, population, and proximity to the ocean. This type of prediction system can be useful in real estate platforms or property evaluation services to assist users in understanding current market trends.


## 📂 Dataset

The dataset used is a CSV file named `housing.csv`, which contains the following features:

- `longitude`: Geographic coordinate
- `latitude`: Geographic coordinate
- `housing_median_age`: Median age of houses in the district
- `total_rooms`: Total number of rooms
- `total_bedrooms`: Total number of bedrooms
- `population`: Population of the district
- `households`: Number of households
- `median_income`: Median income of residents
- `ocean_proximity`: Categorical value indicating location
- `median_house_value`: Target column (renamed as `price`



## 🧼 Data Preprocessing

Key steps:

1. **Rename Target Column**: `median_house_value` → `price`
2. **Handle Missing Values**: Dropped rows with nulls.
3. **One-Hot Encoding**: Categorical variable `ocean_proximity` is encoded using `pd.get_dummies()`.
4. **Correlation Heatmap**: To analyze relationships between numeric features and the target.

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")

#  🧠 MODEL DEVELOPMENT:
We used a Linear Regression model, a basic yet powerful approach for predicting continuous outcomes.

## Steps:
Train-Test Split: 80% training, 20% testing

Model Training: Fit the Linear Regression model on training data

Prediction: Predict prices for the test set

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📊 Evaluation Metrics:

The model is evaluated using the following metrics:

R² Score: Measures how well the model explains variability in the data

RMSE (Root Mean Squared Error): Average difference between predicted and actual values

Example Output:
R² Score: 0.64
RMSE: 68372.48


# 💾 Saving the Model :
The model is saved as a .pkl file using joblib, allowing you to reuse it without retraining:
joblib.dump(model, "models/house_price_model.pkl")


📁 Project Structure


├── housing.csv
├── house_price_prediction.py
├── models/
│   └── house_price_model.pkl
└── README.md

# How to run :
Load the dataset housing.csv
Install required Python packages:
pip install pandas numpy matplotlib seaborn scikit-learn joblib
Run the Python script:
python house_price_prediction.py

# 🚀 Future Improvements:
Add advanced models like Random Forest, XGBoost

Implement feature selection and scaling

Integrate with Flask/FastAPI to serve predictions via API

Add UI for uploading CSV and viewing predictions


# 🙋 Contact
Author: [ POLASANI PURUSHOTHAM REDDY ]
Internship: Next24tech Technology & Services

# OUTPUT : 

![Image](https://github.com/user-attachments/assets/feb9a280-3844-4feb-be69-05b575fbddf3)


![Image](https://github.com/user-attachments/assets/1b9fe273-c1cb-4768-9937-4f0d418fa553)

