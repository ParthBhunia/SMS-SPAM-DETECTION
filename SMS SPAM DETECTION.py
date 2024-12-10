#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:





# In[1]:


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


# In[2]:


df=pd.read_csv(r"C:\Users\Parth Bhunia\Downloads\ML\sms-spam-detection-main\sms-spam-detection-main\sms-spam.csv",encoding='latin1')


# # Data Preprocessing
# 

# In[3]:


print(df.head())


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
df


# In[7]:


df


# In[8]:


df.describe()


# In[9]:


df


# In[10]:


df.isnull().sum()


# In[11]:


df.columns


# In[12]:


df.rename(columns={'ï»¿v1': 'target', 'v2': 'text'}, inplace=True)
df.head()


# In[13]:


df.duplicated().sum()


# In[14]:


df = df.drop_duplicates(keep='first')
df


# In[15]:


encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df


# In[16]:


df.info()


# In[ ]:





# # Applying Logistic Regression

# In[17]:


X = df['text']  # Features (text)
y = df['target']  # Target (ham/spam labels)

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


# Initialize TF-IDF vectorizer with a maximum of 3000 features
vectorizer = TfidfVectorizer(max_features=3000)

# Fit and transform the training data; transform the test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# In[19]:


# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the TF-IDF-transformed training data
model.fit(X_train_tfidf, y_train)


# In[20]:


y_pred = model.predict(X_test_tfidf)


# # Accuracy

# In[21]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


# In[ ]:





# # Confusion Matrix

# In[22]:


# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[23]:


plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()


# In[24]:


class_report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
print("Classification Report:")
print(class_report)


# # Applying Naive Beyes Algorithm

# In[25]:


nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)


# In[26]:


y_pred = nb_model.predict(X_test_tfidf)


# # Result

# In[27]:


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


# In[28]:


print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)


# # Accuracy

# In[29]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


# In[30]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[31]:


plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()


# # Applying Support Vector machine Model

# In[32]:


svm_model = SVC(kernel='linear')  # Linear kernel for text classification
svm_model.fit(X_train_tfidf, y_train)


# In[33]:


y_pred = svm_model.predict(X_test_tfidf)


# In[34]:


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


# # Result

# In[35]:


print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)


# In[36]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[37]:


plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()


# # Applying Random Forest Model

# In[38]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)


# In[39]:


y_pred = rf_model.predict(X_test_tfidf)


# In[40]:


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


# # Result

# In[41]:


print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)


# In[42]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[43]:


plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()


# # COMPARISONS

# In[44]:


pip install matplotlib seaborn


# In[45]:


# Importing required libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Accuracy scores for different models
accuracy_scores = {
    "Logistic Regression": 0.971,
    "SVM": 0.9874,
    "Naive Bayes": 0.9758,
    "Random Forest": 0.9777
}

# Plotting the accuracy comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette='viridis')
plt.title('Accuracy Comparison of Logistic Regression, SVM, Naive Bayes, and Random Forest', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0.9, 1)  # Adjusting y-axis limits for clarity
plt.show()


# In[46]:


# Importing required libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Precision, Recall, and F1-score for different models
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

# Plotting Precision Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=list(precision_scores.keys()), y=list(precision_scores.values()), palette='Blues')
plt.title('Precision Comparison of Logistic Regression, SVM, Naive Bayes, and Random Forest', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.ylim(0.9, 1)  # Adjusting y-axis limits for clarity
plt.show()

# Plotting Recall Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=list(recall_scores.keys()), y=list(recall_scores.values()), palette='Greens')
plt.title('Recall Comparison of Logistic Regression, SVM, Naive Bayes, and Random Forest', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Recall', fontsize=14)
plt.ylim(0.9, 1)  # Adjusting y-axis limits for clarity
plt.show()

# Plotting F1-Score Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()), palette='Purples')
plt.title('F1-Score Comparison of Logistic Regression, SVM, Naive Bayes, and Random Forest', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('F1-Score', fontsize=14)
plt.ylim(0.9, 1)  # Adjusting y-axis limits for clarity
plt.show()


# In[47]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[48]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])  # Ensure df['text'] is preprocessed
y = df['target']


# In[49]:


print(X.shape)


# In[50]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[51]:


from sklearn.tree import DecisionTreeClassifier

# Create the classifier instance
dt_classifier = DecisionTreeClassifier()

# Now fit the classifier to the training data
dt_classifier.fit(X_train, y_train)


# In[52]:


dt_classifier.fit(X_train, y_train)


# In[53]:


y_pred_dt = dt_classifier.predict(X_test)


# In[54]:


accuracy_dt = accuracy_score(y_test, y_pred_dt)
confusion_dt = confusion_matrix(y_test, y_pred_dt)
classification_dt = classification_report(y_test, y_pred_dt)


# In[55]:


print("Decision Tree Classifier Accuracy:", accuracy_dt)
print("Confusion Matrix:\n", confusion_dt)
print("Classification Report:\n", classification_dt)


# In[56]:


from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[57]:


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


# In[58]:


plt.figure(figsize=(8, 6))
sns.heatmap(confusion_dt, annot=True, fmt='d', cmap='Blues', 
            xticklabels=encoder.classes_, 
            yticklabels=encoder.classes_)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[59]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



# Encode the target variable
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df = df.drop_duplicates()


# List of models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Classifier": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

# Dictionary to store accuracy results
accuracy_results = {}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results[model_name] = accuracy

# Create a DataFrame for visualization
results_df = pd.DataFrame(list(accuracy_results.items()), columns=['Model', 'Accuracy'])

# Plotting the comparison chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=results_df, palette='viridis')
plt.title('Model Comparison: Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.xlim(0, 1)  # Set limits for the x-axis
plt.show()


# In[60]:


import joblib
# Save your trained model to a file
joblib.dump(svm_model, 'svm_model.pkl')


# In[61]:


joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')


# In[ ]:




