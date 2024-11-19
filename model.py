
import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv(r"C:\Users\Parth Bhunia\OneDrive\Desktop\SPAM\spam.csv",encoding='latin1')
print(df.head())





vectorizer = TfidfVectorizer(max_features=3000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)



accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()


class_report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
print("Classification Report:")
print(class_report)




nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)


y_pred = nb_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()

svm_model = SVC(kernel='linear')  # Linear kernel for text classification
svm_model.fit(X_train_tfidf, y_train)


y_pred = svm_model.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)


y_pred = rf_model.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()



accuracy_scores = {
    "Logistic Regression": 0.971,
    "SVM": 0.9874,
    "Naive Bayes": 0.9758,
    "Random Forest": 0.9777
}

plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette='viridis')
plt.title('Accuracy Comparison of Logistic Regression, SVM, Naive Bayes, and Random Forest', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0.9, 1)  # Adjusting y-axis limits for clarity
plt.show()



precision_scores = {
    "Logistic Regression":0.97 ,
    "SVM": 0.99,
    "Naive Bayes": 0.98,
    "Random Forest": 0.98
}

recall_scores = {
    "Logistic Regression": 0.97,
    "SVM": 0.99,
    "Naive Bayes": 0.98,
    "Random Forest": 0.98
}

f1_scores = {
    "Logistic Regression": 0.97,
    "SVM": 0.99,
    "Naive Bayes": 0.97,
    "Random Forest": 0.98
}

plt.figure(figsize=(10, 6))
sns.barplot(x=list(precision_scores.keys()), y=list(precision_scores.values()), palette='Blues')
plt.title('Precision Comparison of Logistic Regression, SVM, Naive Bayes, and Random Forest', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.ylim(0.9, 1)  # Adjusting y-axis limits for clarity
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=list(recall_scores.keys()), y=list(recall_scores.values()), palette='Greens')
plt.title('Recall Comparison of Logistic Regression, SVM, Naive Bayes, and Random Forest', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Recall', fontsize=14)
plt.ylim(0.9, 1)  # Adjusting y-axis limits for clarity
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()), palette='Purples')
plt.title('F1-Score Comparison of Logistic Regression, SVM, Naive Bayes, and Random Forest', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('F1-Score', fontsize=14)
plt.ylim(0.9, 1)  # Adjusting y-axis limits for clarity
plt.show()


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])  # Ensure df['text'] is preprocessed
y = df['target']



print(X.shape)





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





dt_classifier = DecisionTreeClassifier()

dt_classifier.fit(X_train, y_train)




dt_classifier.fit(X_train, y_train)


y_pred_dt = dt_classifier.predict(X_test)


accuracy_dt = accuracy_score(y_test, y_pred_dt)
confusion_dt = confusion_matrix(y_test, y_pred_dt)
classification_dt = classification_report(y_test, y_pred_dt)


print("Decision Tree Classifier Accuracy:", accuracy_dt)
print("Confusion Matrix:\n", confusion_dt)
print("Classification Report:\n", classification_dt)





plt.figure(figsize=(100, 100))  # Increase figure size for better visibility
plot_tree(dt_classifier, 
          filled=True, 
          feature_names=vectorizer.get_feature_names_out().tolist(),  # Convert to list
          class_names=encoder.classes_.tolist(),  # Convert to list
          rounded=True,
          fontsize=10,
          precision=2)  # Show precision in the tree nodes
plt.title("Decision Tree Visualization (Max Depth = 2)")
plt.show()



plt.figure(figsize=(8, 6))
sns.heatmap(confusion_dt, annot=True, fmt='d', cmap='Blues', 
            xticklabels=encoder.classes_, 
            yticklabels=encoder.classes_)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()






encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df = df.drop_duplicates()



models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Classifier": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}


accuracy_results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results[model_name] = accuracy


results_df = pd.DataFrame(list(accuracy_results.items()), columns=['Model', 'Accuracy'])


plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=results_df, palette='viridis')
plt.title('Model Comparison: Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.xlim(0, 1)  # Set limits for the x-axis
plt.show()







