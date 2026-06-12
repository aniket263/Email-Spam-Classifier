import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page Config
st.set_page_config(
    page_title="AI-Powered Email Security Analyzer",
    page_icon="📧",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Main Card */
.main {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(15px);
    padding: 20px;
    border-radius: 20px;
}

/* Title */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #e0e0e0;
    font-size: 18px;
    margin-bottom: 20px;
}

/* Text Area */
.stTextArea textarea {
    border-radius: 12px !important;
    border: 2px solid #cccccc !important;
}

/* Button */
.stButton button {
    width: 100%;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 18px;
    font-weight: bold;
}

.stButton button:hover {
    transform: scale(1.02);
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 30px;
    color: gray;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="title">📧 AI-Powered Email Security Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Detect whether an Email or SMS message is Spam or Safe using Machine Learning</div>',
    unsafe_allow_html=True
)

# Input Box
message = st.text_area(
    "✍️ Enter your message",
    height=180,
    placeholder="Type or paste your Email/SMS content here..."
)

# Character Counter
st.caption(f"Characters: {len(message)}")

# Prediction
if st.button("🔍 Analyze Message"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)

        if prediction[0] == 1:
            st.error("🚨 SPAM MESSAGE DETECTED")
            st.progress(90)
            st.write("Spam Probability: 90%")
        else:
            st.success("✅ SAFE MESSAGE")
            st.progress(10)
            st.write("Spam Probability: 10%")

# Footer
st.markdown(
    '<div class="footer"> Developed by Aniket Sahu</div>',
    unsafe_allow_html=True
)