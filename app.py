import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------------------
# APP CONFIG
# -------------------------------
st.set_page_config(page_title="📦 E-Commerce Delivery Predictor", layout="wide")
st.title("📦 E-Commerce Delivery Prediction App")

MODEL_FILE = "decision_tree_model.joblib"
DATA_FILE = "data.csv"

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def train_model():
    """Train and save model from data.csv"""
    if not os.path.exists(DATA_FILE):
        st.error("❌ 'data.csv' file not found in the current folder!")
        return None

    try:
        df = pd.read_csv(DATA_FILE)
        X = df.drop("OnTimeDelivery", axis=1)
        y = df["OnTimeDelivery"]

        numeric_features = [
            "Customer_care_calls",
            "Customer_rating",
            "Cost_of_the_Product",
            "Prior_purchases",
            "Discount_offered",
            "Weight_in_gms"
        ]
        categorical_features = ["Warehouse_block", "Mode_of_Shipment", "Gender"]
        ordinal_features = ["Product_importance"]
        importance_levels = [["low", "medium", "high"]]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("ord", OrdinalEncoder(categories=importance_levels), ordinal_features)
            ]
        )

        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(class_weight="balanced", max_depth=10, random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILE)
        st.success("✅ Model trained and saved successfully!")
        return model
    except Exception as e:
        st.error(f"⚠️ Training failed: {e}")
        return None


def load_model():
    """Load trained model"""
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        return joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {e}")
        return None


def predict(model, data):
    """Make prediction"""
    proba = model.predict_proba(data)[0][1]
    return proba

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("⚙️ Control Panel")

model = load_model()
if model:
    st.sidebar.success("✅ Model Loaded")
else:
    st.sidebar.warning("⚠️ No trained model found")

if st.sidebar.button("🚀 Train / Retrain Model"):
    model = train_model()

st.sidebar.markdown("---")
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.01)

# -------------------------------
# MAIN APP INTERFACE
# -------------------------------
st.subheader("🧠 Enter Product and Customer Details")

col1, col2 = st.columns(2)

with col1:
    warehouse_block = st.selectbox("Warehouse Block", ['A', 'B', 'C', 'D', 'F'])
    mode = st.selectbox("Mode of Shipment", ['Ship', 'Flight', 'Road'])
    calls = st.slider("Customer Care Calls", 2, 7, 4)
    rating = st.select_slider("Customer Rating", [1, 2, 3, 4, 5], value=3)
    cost = st.number_input("Cost of Product (USD)", 50, 300, 150)

with col2:
    purchases = st.slider("Prior Purchases", 2, 10, 4)
    importance = st.radio("Product Importance", ['low', 'medium', 'high'], horizontal=True)
    gender = st.radio("Customer Gender", ['F', 'M'], horizontal=True)
    discount = st.number_input("Discount Offered (%)", 0, 65, 10)
    weight = st.number_input("Weight in Grams", 500, 8000, 2000)

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("🔍 Predict Delivery Status"):
    if model is None:
        st.error("❌ Please train or load a model first!")
    else:
        try:
            input_df = pd.DataFrame({
                'Warehouse_block': [warehouse_block],
                'Mode_of_Shipment': [mode],
                'Customer_care_calls': [calls],
                'Customer_rating': [rating],
                'Cost_of_the_Product': [cost],
                'Prior_purchases': [purchases],
                'Product_importance': [importance],
                'Gender': [gender],
                'Discount_offered': [discount],
                'Weight_in_gms': [weight]
            })

            p_delay = predict(model, input_df)
            result = 1 if p_delay >= threshold else 0

            if result == 0:
                st.success(f"✅ Product will *Reach On Time* (0)\n**Delay Probability:** {p_delay:.2f}")
                st.balloons()
            else:
                st.error(f"⚠️ Product will *NOT Reach On Time* (1)\n**Delay Probability:** {p_delay:.2f}")

            st.info("🧭 Adjust threshold from sidebar to control prediction sensitivity.")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
