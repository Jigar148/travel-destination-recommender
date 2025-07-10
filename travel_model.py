import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------
# Load the data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("travel_data.csv")
    return df

df = load_data()

# -------------------------------
# Preprocess the data
# -------------------------------
# Drop 'budget_display' as it's redundant
df = df.drop(columns=["budget_display"])

# Encode categorical variables
label_encoders = {}
for column in df.columns:
    if df[column].dtype == "object":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Split into features and target
X = df.drop("destination", axis=1)
y = df["destination"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Train the model
# -------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🌍 Travel Destination Predictor")

st.write("Fill in your preferences below to predict a suitable travel destination.")

# Input fields (reverse label encoding to get original values)
def select_input(label, column):
    options = label_encoders[column].classes_
    return st.selectbox(label, options)

budget = select_input("💰 Budget Currency", "budget")
climate = select_input("🌤️ Climate", "climate")
season = select_input("🌧️ Season", "season")
travel_type = select_input("✈️ Travel Type", "travel_type")
group_type = select_input("👨‍👩‍👧‍👦 Group Type", "group_type")
duration = select_input("🕒 Duration", "duration")
continent = select_input("🌎 Continent", "continent")
amount_usd = st.number_input("💵 Estimated Budget in USD", min_value=0)

# Predict button
if st.button("Predict Destination"):
    # Encode the inputs
    input_data = {
        "budget": label_encoders["budget"].transform([budget])[0],
        "climate": label_encoders["climate"].transform([climate])[0],
        "season": label_encoders["season"].transform([season])[0],
        "travel_type": label_encoders["travel_type"].transform([travel_type])[0],
        "group_type": label_encoders["group_type"].transform([group_type])[0],
        "duration": label_encoders["duration"].transform([duration])[0],
        "continent": label_encoders["continent"].transform([continent])[0],
        "amount_usd": amount_usd
    }

    input_df = pd.DataFrame([input_data])
    pred_encoded = model.predict(input_df)[0]
    pred_destination = label_encoders["destination"].inverse_transform([pred_encoded])[0]

    st.success(f"🏖️ Recommended Destination: **{pred_destination}**")

