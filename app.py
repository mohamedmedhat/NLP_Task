import streamlit as st
import sklearn
import helper
import pickle
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

model=pickle.load(open("models/model.pkl",'rb'))
vectorizer=pickle.load(open("models/vectorizer.pkl",'rb'))

st.title("Sentiment Analysis App")
st.markdown("""
    Welcome to the Sentiment Analysis tool!  
    Enter a review below, and I will predict if the sentiment is positive or negative.
""")

st.subheader("Your Review:")
text = st.text_input("Please enter your review:")

state = st.button("Predict Sentiment")

if state:
    token = helper.preprocessing_step(text)
    vectorized_data = vectorizer.transform([token])
    prediction = model.predict(vectorized_data)

    if prediction == 1:
        sentiment = "Positive 😄"
        st.success(f"The sentiment of your review is: {sentiment}")
    else:
        sentiment = "Negative 😞"
        st.error(f"The sentiment of your review is: {sentiment}")

st.markdown("---")

st.markdown("""
    <footer style="text-align:center; font-size:14px;">
        Created with ❤️ by Your Name
    </footer>
    """, unsafe_allow_html=True)
