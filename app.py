import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📩"
)

st.title("📩 Email Spam Classifier")
st.write("Detect whether an email or SMS message is Spam or Ham using Machine Learning.")

message = st.text_area("Enter your message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        message_vector = vectorizer.transform([message])

        prediction = model.predict(message_vector)

        if prediction[0] == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Ham (Normal Message)")