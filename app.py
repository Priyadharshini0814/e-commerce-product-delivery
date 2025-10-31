import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------------------
# APP CONFIG
# -------------------------------
st.set_page_config(
    page_title="🛒 E-Commerce Sales & Delivery Predictor",
    layout="wide",
    page_icon="🛍️"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
h1, h2, h3 {
    color: #003366;
}
.stButton > button {
    background-color: #0078D7;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    transition: 0.3s;
}
.stButton > button:hover {
    background-color: #005ea6;
}
.metric-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# FILES & MODELS
# -------------------------------
MODEL_FILE_DELIVERY = "decision_tree_model.joblib"
MODEL_FILE_SALES = "sales_model.joblib"
DATA_FILE = "data.csv"

# -------------------------------
# TRAIN MODELS
# -------------------------------
def train_delivery_model():
    if not os.path.exists(DATA_FILE):
        st.error("❌ data.csv not found!")
        return None
    try:
        df = pd.read_csv(DATA_FILE)
        X = df.drop("OnTimeDelivery", axis=1)
        y = df["OnTimeDelivery"]

        numeric = ["Customer_care_calls", "Customer_rating", "Cost_of_the_Product",
                   "Prior_purchases", "Discount_offered", "Weight_in_gms"]
        categorical = ["Warehouse_block", "Mode_of_Shipment", "Gender"]
        ordinal = ["Product_importance"]
        levels = [["low", "medium", "high"]]

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("ord", OrdinalEncoder(categories=levels), ordinal)
        ])

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(max_depth=10, random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILE_DELIVERY)
        st.success("✅ Delivery Model trained successfully!")
        return model
    except Exception as e:
        st.error(f"⚠️ Training failed: {e}")
        return None


def train_sales_model():
    if not os.path.exists(DATA_FILE):
        st.error("❌ data.csv not found!")
        return None
    try:
        df = pd.read_csv(DATA_FILE)
        if "Sales" not in df.columns:
            df["Sales"] = df["Cost_of_the_Product"] * (100 - df["Discount_offered"]) / 100 * df["Prior_purchases"]

        X = df.drop("Sales", axis=1)
        y = df["Sales"]

        numeric = ["Customer_care_calls", "Customer_rating", "Cost_of_the_Product",
                   "Prior_purchases", "Discount_offered", "Weight_in_gms"]
        categorical = ["Warehouse_block", "Mode_of_Shipment", "Gender"]
        ordinal = ["Product_importance"]
        levels = [["low", "medium", "high"]]

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("ord", OrdinalEncoder(categories=levels), ordinal)
        ])

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", DecisionTreeRegressor(max_depth=10, random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILE_SALES)
        st.success("✅ Sales Model trained successfully!")
        return model
    except Exception as e:
        st.error(f"⚠️ Training failed: {e}")
        return None


# -------------------------------
# LOAD MODEL FUNCTION
# -------------------------------
def load_model(file):
    if not os.path.exists(file):
        return None
    try:
        return joblib.load(file)
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        return None


# -------------------------------
# MAIN INTERFACE
# -------------------------------
st.title("🛍️ E-Commerce Sales & Delivery Predictor")
st.markdown("### Predict whether your order will arrive on time and estimate expected sales in ₹ INR!")

col1, col2 = st.columns(2)

with col1:
    warehouse_block = st.selectbox("Warehouse Block", ['A', 'B', 'C', 'D', 'F'])
    mode = st.selectbox("Mode of Shipment", ['Ship', 'Flight', 'Road'])
    calls = st.slider("Customer Care Calls", 2, 7, 4)
    rating = st.select_slider("Customer Rating", [1, 2, 3, 4, 5], value=3)
    cost = st.number_input("Cost of Product (₹)", 100, 25000, 5000)

with col2:
    purchases = st.slider("Prior Purchases", 1, 10, 4)
    importance = st.radio("Product Importance", ['low', 'medium', 'high'], horizontal=True)
    gender = st.radio("Customer Gender", ['F', 'M'], horizontal=True)
    discount = st.number_input("Discount Offered (%)", 0, 65, 10)
    weight = st.number_input("Weight (in grams)", 100, 10000, 2000)

# -------------------------------
# PREDICTIONS
# -------------------------------
delivery_model = load_model(MODEL_FILE_DELIVERY)
sales_model = load_model(MODEL_FILE_SALES)

threshold = st.slider("Delivery Decision Threshold", 0.1, 0.9, 0.5, 0.01)

colA, colB = st.columns(2)
with colA:
    if st.button("🚚 Predict Delivery Status"):
        if delivery_model is None:
            st.error("❌ Please train or load the delivery model first!")
        else:
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
            prob = delivery_model.predict_proba(input_df)[0][1]
            result = "🟢 On Time" if prob < threshold else "🔴 Delayed"
            with st.container():
                st.markdown(
                    f"<div class='metric-box'><h3>Delivery Prediction:</h3><h2>{result}</h2><p>Delay Probability: {prob:.2f}</p></div>",
                    unsafe_allow_html=True,
                )

with colB:
    if st.button("💰 Predict Sales (₹ INR)"):
        if sales_model is None:
            st.error("❌ Please train or load the sales model first!")
        else:
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
            predicted_sales = sales_model.predict(input_df)[0]
            with st.container():
                st.markdown(
                    f"<div class='metric-box'><h3>Predicted Sales:</h3><h2>₹ {predicted_sales:,.2f}</h2></div>",
                    unsafe_allow_html=True,
                )

st.markdown("---")
st.info("💡 Tip: Train or retrain your models whenever you update your dataset for best results.")

if st.button("🔁 Train Both Models"):
    train_delivery_model()
    train_sales_model()
