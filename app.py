import streamlit as st
import pickle
import pandas as pd

# Load Saved Model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🌸 Iris Flower Classification App")
st.write("Enter the flower measurements below:")

# User Input
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.1f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.1f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.1f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.1f")

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted Flower: {prediction[0]}")