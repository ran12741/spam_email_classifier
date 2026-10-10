import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("email.csv")

# Remove duplicate rows
data = data.drop_duplicates()

# Clean category values
data["Category"] = data["Category"].astype(str).str.strip().str.lower()

# Convert ham/spam into numbers
data["Category"] = data["Category"].map({
    "ham": 0,
    "spam": 1
})

# Check if any NaN was created
print("Missing labels after conversion:", data["Category"].isna().sum())

# Remove any rows with invalid labels, if present
data = data.dropna(subset=["Category"])

# Input and target
X = data["Message"]
y = data["Category"].astype(int)

# Convert email text into numerical features
vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(X)

print("Number of messages:", len(X))
print("TF-IDF matrix shape:", X_tfidf.shape)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.naive_bayes import MultinomialNB

# Create the Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")
from sklearn.metrics import accuracy_score, classification_report

# Predict the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)

# Detailed performance report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Spam Email Classification - Confusion Matrix")
plt.show()
# Take a new email from the user
new_email = input("\nEnter an email message: ")

# Convert the new email into TF-IDF features
new_email_tfidf = vectorizer.transform([new_email])

# Predict
prediction = model.predict(new_email_tfidf)

# Display result
if prediction[0] == 1:
    print("Prediction: SPAM")
else:
    print("Prediction: HAM")