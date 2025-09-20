import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

print("--- Starting Model Training ---")

# 1. Load the CLEANED dataset
df = pd.read_csv('spam_cleaned.csv')

# Drop rows with missing messages just in case
df.dropna(subset=['message'], inplace=True)

# 2. Split data into features (X) and labels (y)
X = df['message']
y = df['label']

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 7. Save the trained model AND the vectorizer
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("--- Model and Vectorizer saved successfully! ---")
