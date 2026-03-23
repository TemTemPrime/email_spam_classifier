import streamlit as st
import pickle
import pandas as pd
import joblib


@st.cache_resource
def load_model():
    model = joblib.load("matcher_assets.pkl")  
    return model
assets  = load_model()
model = assets['model']
vectorizer = assets['vectorizer']
df = assets['original_df']


st.title("email spam classifier")
st.write("paste email here")
email = st.text_input("Insert Email")
if st.button("predict"):
    if email.strip() == "":
        st.warning("Please enter an email message.")
    else:
        vector_email = vectorizer.transform([email])
        prediction = model.predict([vector_email])
        if prediction == 1:
            st.error("🚨 This email is SPAM")
        else:
            st.success("✅ This email is NOT Spam")
            