import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Select useful columns

df = df[['tenure', 'MonthlyCharges', 'SeniorCitizen', 'Churn']]

# Convert Churn values

encoder = LabelEncoder()

df['Churn'] = encoder.fit_transform(df['Churn'])

# Features and target

X = df[['tenure', 'MonthlyCharges', 'SeniorCitizen']]

y = df['Churn']

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model

model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions

predictions = model.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save model

joblib.dump(model, 'churn_model.pkl')

print("Real Dataset Model Trained Successfully")