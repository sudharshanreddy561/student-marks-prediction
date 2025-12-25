# Student Marks Prediction using Machine Learning
# Author: Sudharshan

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("students.csv")

X = data[["hours_studied"]]   # Feature
y = data["marks"]             # Target

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
hours = [[5]]
predicted_marks = model.predict(hours)

print(f"Predicted marks for {hours[0][0]} hours study: {predicted_marks[0]:.2f}")

# Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Hours Studied vs Marks")
plt.show()
