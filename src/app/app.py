import streamlit as st
import requests
import json

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
    

def check_api_health():
    response = requests.get("http://api:8086/latest")
    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        if data["health_check"] == "OK":
            return True
        else:
            return False
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return False
    


def main():   
    st.set_page_config(page_title="Email Spam Classifier", page_icon=":email:", layout="wide")
    
    if check_api_health():
        st.success("API is healthy")
    else:
        st.error("API is not healthy. Please check the API server.")

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
                    for word in preprocessed_email.split():
                        st.write(f"`{word}` ", end="")
                else:
                    st.error("Error: Unable to classify the email. Please try again later.")

if __name__ == "__main__":
    main()
