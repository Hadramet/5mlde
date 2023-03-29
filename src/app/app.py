import streamlit as st
import requests
import json

# Define the email spam classification function
def classify_email(email):
    payload = json.dumps({"text": email})
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://api:8086/classify", data=payload, headers=headers)
    
    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        return data["label"], data["test"]["text"]["0"]
    
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None, None

# Main app function
def main():
    st.set_page_config(page_title="Email Spam Classifier", page_icon=":email:", layout="wide")

    # UI elements
    st.title("Email Spam Classifier :email:")
    email_input = st.text_area("Enter the email content:")
    classify_button = st.button("Classify Email")

    if classify_button:
        if email_input.strip() == "":
            st.error("Please enter the email content.")
        else:
            with st.spinner("Classifying..."):
                label, preprocessed_email = classify_email(email_input)
                if label is not None and preprocessed_email is not None:
                    if label == 1:
                        st.warning("This email is spam.")
                    else:
                        st.success("This email is not spam.")
                    
                    st.subheader("Preprocessed Email:")
                    words = preprocessed_email.split()
                    for word in words:
                        st.markdown(f'<span style="background-color:#f0f0f0; border-radius: 5px; padding: 4px 6px; margin: 2px;">{word}</span>', unsafe_allow_html=True)
                else:
                    st.error("Error: Unable to classify the email. Please try again later.")

if __name__ == "__main__":
    main()