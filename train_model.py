import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Sample training data
data = {
    'tenure': [1, 2, 3, 10, 12, 15],
    'MonthlyCharges': [70, 90, 80, 40, 30, 20],
    'Churn': [1, 1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

X = df[['tenure', 'MonthlyCharges']]
y = df['Churn']

model = DecisionTreeClassifier()

model.fit(X, y)

pickle.dump(model, open('churn_model.pkl', 'wb'))

print("Model trained successfully")