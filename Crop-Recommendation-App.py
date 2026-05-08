import streamlit as st
import pickle 
import numpy as np
st.title("CROP RECOMMENDATION SYSTEM") 
#st.write("This is a simple crop recommendation system that recommends the best crop to grow based on the soil and weather conditions.")
#st.slider("Nitrogen")
model = pickle.load(open("model.pkl", "rb"))
sepal_length1 = st.slider("N", 0.0, 140.0)
sepal_length2 = st.slider("P", 5.0, 85.0)
sepal_length3 = st.slider("K", 5.0, 85.0)
sepal_length4 = st.slider("temperature", 15.0, 36.0)
sepal_length5 = st.slider("humidity", 14.0, 100.0)
sepal_length6 = st.slider("ph", 4.0, 10.0)
sepal_length7 = st.slider("rainfall", 21.0, 299.0)
if st.button("Predict"):
    features = np.array([[sepal_length1, sepal_length2, sepal_length3, sepal_length4, sepal_length5, sepal_length6, sepal_length7]])
    prediction = model.predict(features)
    st.success(f"Predicted Class: {prediction[0]}")