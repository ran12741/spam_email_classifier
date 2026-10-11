import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")

st.title("📧 Email Spam Classifier")
st.write("Enter an email message to check whether it is HAM or SPAM.")

@st.cache_resource
def train_model():
data = pd.read_csv("email.csv")
data = data.dropna(subset=["Category", "Message"]).drop_duplicates()
data["Category"] = data["Category"].astype(str).str.strip().str.lower()
data = data[data["Category"].isin(["ham", "spam"])]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["Message"].astype(str))
y = data["Category"]

model = MultinomialNB()
model.fit(X, y)

return model, vectorizer

try:
model, vectorizer = train_model()

email = st.text_area(
    "Enter your email message:",
    placeholder="Type or paste your email here..."
)

if st.button("Check Email"):
    if email.strip():
        prediction = model.predict(vectorizer.transform([email]))[0]

        if prediction == "spam":
            st.error("🚨 This email is SPAM!")
        else:
            st.success("✅ This email is HAM (Not Spam).")
    else:
        st.warning("Please enter an email message.")

except Exception as e:
st.error("App error: Please check email.csv and its column names.")
st.exception(e)
