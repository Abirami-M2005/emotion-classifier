import streamlit as st
import joblib

# Load the saved models and tools
clf = joblib.load("decision_tree_model.pkl")  # or use logistic_regression_model.pkl
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# App title and description
st.title("Emotion Classification from Tweets")
st.write("Enter a tweet below, and the model will predict the emotion it expresses.")

# User input
user_input = st.text_area("Enter Tweet Text")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Transform input using TF-IDF vectorizer
        X_input = vectorizer.transform([user_input])
        
        # Make prediction
        prediction = clf.predict(X_input)
        predicted_emotion = label_encoder.inverse_transform(prediction)[0]
        
        # Display result
        st.success(f"Predicted Emotion: **{predicted_emotion}**")
