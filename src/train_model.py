import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_excel("data/Telco_customer_churn.xlsx")

# Select features
features = ["Tenure Months", "Monthly Charges"]

X = df[features]
y = df["Churn Value"]

# Target
y = df["Churn Value"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained successfully and saved as model.pkl")

model_data = {
    "model": model,
    "features": features
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)