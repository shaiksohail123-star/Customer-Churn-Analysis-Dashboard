import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.title("Customer Churn Analysis Dashboard")

# LOAD DATASET FIRST
df = pd.read_excel("data/Telco_customer_churn.xlsx")

# KPI CARDS
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Churned Customers", df["Churn Value"].sum())
col3.metric("Churn Rate", f"{df['Churn Value'].mean()*100:.2f}%")

# LOAD TRAINED MODEL
model_data = pickle.load(open("model.pkl", "rb"))
model = model_data["model"]

# Charts
st.subheader("Churn by Contract Type")
contract_churn = pd.crosstab(df["Contract"], df["Churn Label"])
st.bar_chart(contract_churn)

st.subheader("Monthly Charges vs Churn")
fig, ax = plt.subplots()
sns.boxplot(x="Churn Label", y="Monthly Charges", data=df, ax=ax)
st.pyplot(fig)

st.subheader("Churn Distribution")
churn_counts = df["Churn Label"].value_counts()
st.bar_chart(churn_counts)

# Feature Importance
st.subheader("Feature Importance")
importance = pd.Series(model.feature_importances_,
                       index=["Tenure Months", "Monthly Charges"])
st.bar_chart(importance)

# Prediction
st.subheader("Predict Customer Churn")

tenure = st.slider("Tenure Months", 1, 72, 12)
monthly_charge = st.number_input("Monthly Charges", 0, 200, 70)

if st.button("Predict Churn"):
    prediction = model.predict([[tenure, monthly_charge]])

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")