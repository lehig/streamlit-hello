import streamlit as st
import joblib

st.title("Gender Prediction")

# load the model
@st.cache_resource
def load_model():
    return joblib.load("naive_bayes_gender_model_US.py")

model = load_model()

st.subheader("Enter measurements")

col1, col2 = st.columns(2)

with col1:
    feet = st.number_input("Height (feet)", min_value=1, max_value=8, value=5, step=1)

with col2:
    inches = st.number_input("Height (inches)", min_value=0, max_value=11, value=6, step=1)


