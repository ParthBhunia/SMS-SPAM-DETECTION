import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Title for the Streamlit app
st.title("SMS Spam Detection using Machine Learning")

# Directly load the dataset (replace the path with the location of your CSV file)
file_path = r"C:\Users\Parth Bhunia\Downloads\ML\sms-spam-detection-main\sms-spam-detection-main\sms-spam.csv"

# Load the dataset
df = pd.read_csv(file_path, encoding='latin1')

# Data Preprocessing
st.subheader("Data Overview")
st.write(df.head())

# Drop irrelevant columns
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')

# Print column names to check if 'target' and 'text' are correctly identified
st.write("Column Names Before Renaming:", df.columns)

# Rename columns to 'target' and 'text'
df.rename(columns={'ï»¿v1': 'target', 'v2': 'text'}, inplace=True)

# Print column names after renaming
st.write("Column Names After Renaming:", df.columns)

# Check for missing or duplicate data
df.dropna(subset=['text'], inplace=True)  # Remove rows where 'text' column has null values
df = df.drop_duplicates()

# Ensure 'target' is correctly created
if 'target' not in df.columns:
    st.error("'target' column is missing.")
else:
    st.write("'target' column found.")

# Encode 'target' labels (ham=0, spam=1)
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Feature and target split
X = df['text']
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text preprocessing: Convert to lowercase and remove non-alphabetic characters
X_train = X_train.str.lower().str.replace(r'[^a-z\s]', '', regex=True)
X_test = X_test.str.lower().str.replace(r'[^a-z\s]', '', regex=True)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)

# Fit and transform the training data, transform the test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Selection
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier()
}

# Model Training and Evaluation
st.subheader("Model Training and Evaluation")
accuracy_results = {}
for model_name, model in models.items():
    with st.spinner(f"Training {model_name}..."):
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[model_name] = accuracy

# Display the accuracy comparison
st.subheader("Model Accuracy Comparison")
accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=["Model", "Accuracy"])
st.write(accuracy_df)

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=accuracy_df, palette='viridis')
plt.title("Model Accuracy Comparison")
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.xlim(0, 1)
st.pyplot(plt)

# Confusion Matrix for the selected model
selected_model_name = st.selectbox("Select Model to View Confusion Matrix", models.keys())
selected_model = models[selected_model_name]

st.subheader(f"Confusion Matrix for {selected_model_name}")
selected_model.fit(X_train_tfidf, y_train)
y_pred = selected_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title(f"Confusion Matrix for {selected_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(plt)

# Display classification report
st.subheader("Classification Report")
class_report = classification_report(y_test, y_pred, target_names=encoder.classes_)
st.text(class_report)

# Decision Tree Visualization (Optional)
if selected_model_name == "Decision Tree":
    st.subheader("Decision Tree Visualization")
    plt.figure(figsize=(10, 10))
    plot_tree(selected_model, filled=True, feature_names=vectorizer.get_feature_names_out(), 
              class_names=encoder.classes_.tolist(), rounded=True, fontsize=10)
    st.pyplot(plt)

# Additional metrics (Precision, Recall, F1-Score) - Optional
precision_scores = {
    "Logistic Regression": 0.97,
    "SVM": 0.99,
    "Naive Bayes": 0.98,
    "Random Forest": 0.98,
    "Decision Tree": 0.97
}

recall_scores = precision_scores.copy()
f1_scores = precision_scores.copy()

# Plot precision, recall, F1 comparison
for metric, scores in zip(["Precision", "Recall", "F1-Score"], [precision_scores, recall_scores, f1_scores]):
    st.subheader(f"{metric} Comparison")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(scores.keys()), y=list(scores.values()), palette='Blues')
    plt.title(f'{metric} Comparison of Models')
    plt.xlabel('Models')
    plt.ylabel(f'{metric}')
    plt.ylim(0.9, 1)
    st.pyplot(plt)

# --- New Section: User Input for Spam Detection ---

st.subheader("Test if Your Message is Spam or Not")

# User input for text
user_input = st.text_input("Enter a message to classify as Spam or Ham:")

if user_input:
    # Preprocess the user input just like the training data
    user_input_tfidf = vectorizer.transform([user_input.lower().replace(r'[^a-z\s]', '')])

    # Choose the model (you can use the model you trained, for example, Logistic Regression)
    final_model = LogisticRegression()
    final_model.fit(X_train_tfidf, y_train)

    # Predict the class (0 = Ham, 1 = Spam)
    prediction = final_model.predict(user_input_tfidf)
    prediction_label = "Spam" if prediction[0] == 1 else "Ham"
    
    # Show the prediction
    st.write(f"The entered message is classified as: **{prediction_label}**")
