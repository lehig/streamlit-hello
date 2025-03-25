import streamlit as st
import joblib
import pandas as pd

st.title("Gender Prediction")

# load the model
@st.cache_resource
def load_model():
    return joblib.load("naive_bayes_gender_model_US.pkl")

model = load_model()

st.subheader("Enter measurements")

col1, col2 = st.columns(2)

with col1:
    feet = st.number_input("Height (feet)", min_value=1, max_value=8, value=5, step=1)

with col2:
    inches = st.number_input("Height (inches)", min_value=0, max_value=11, value=6, step=1)


height = feet + (12/inches)

normalized_shoe_size = st.number_input("Shoe size US show", min_value = 1.0, max_value=24.0, value=9.5, step=0.5)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "height": [height],
        "shoe_size": [normalized_shoe_size]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction result: ")

    st.write(f'Prediction: {prediction}')
    st.write(f"Probability: {probability}")