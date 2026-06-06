import pandas as pd
import matplotlib.pyplot as plt

# Load dataset

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Count churn values

churn_counts = df['Churn'].value_counts()

# Create graph

plt.figure(figsize=(6, 6))

plt.pie(
    churn_counts,
    labels=['No Churn', 'Churn'],
    autopct='%1.1f%%'
)

plt.title('Customer Churn Distribution')

# Save graph image

plt.savefig('static/churn_chart.png')

print("Chart Created Successfully")