import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# Load model and features
model = joblib.load('iphone_price_model.pkl')
features = joblib.load('model_feature.pkl')

# ---------- Page Config ----------
st.set_page_config(
    page_title="📱 iPhone Price Predictor",
    page_icon="📱",
    layout="centered"
)

# ---------- Header Section ----------
st.markdown("""
    <h1 style='text-align: center; color: #d63384;'>📱 iPhone Price Predictor</h1>
    <p style='text-align: center; font-size: 18px;'>Estimate the selling price of an iPhone based on its specifications.</p>
    <hr>
""", unsafe_allow_html=True)

# ---------- Sidebar Info ----------
with st.sidebar:
    st.header("📊 Input Specifications")
    Mrp = st.slider('MRP (₹)', min_value=10000, max_value=200000, step=1000, value=60000)
    discount = st.slider('Discount Percentage (%)', min_value=0.0, max_value=50.0, step=0.5, value=10.0)
    ratings = st.number_input('Number Of Ratings', min_value=0, max_value=10000, step=100, value=2000)
    reviews = st.number_input('Number Of Reviews', min_value=0, max_value=5000, step=50, value=500)
    star_rating = st.slider('Star Rating', min_value=0.0, max_value=5.0, step=0.1, value=4.5)
    ram = st.selectbox('RAM (GB)', options=[4, 6, 8, 12, 16], index=2)

# ---------- Main Prediction ----------
if st.button("🔮 Predict iPhone Price"):
    input_df = pd.DataFrame([[Mrp, discount, ratings, reviews, star_rating, ram]], columns=features)
    prediction = model.predict(input_df)[0]
    st.markdown(f"""
        <div style="background-color:#f8f9fa;padding:20px;border-radius:10px;margin-top:20px;">
            <h3 style='text-align: center; color: #20c997;'>📦 Estimated Sale Price: ₹{int(prediction):,}</h3>
        </div>
    """, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; color: gray;'>
        Built with ❤️ by <b>Roshan Patil</b> | Streamlit + ML
    </p>
""", unsafe_allow_html=True)

