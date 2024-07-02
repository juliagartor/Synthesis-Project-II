import os
import re
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
unique_labels = ["Academic transcripts","Bank Account Statements","Free writings","Ids, birth & marriage certificates","University Degrees"]


# Preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'\W', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords and stem
    stop_words = set(stopwords.words('spanish'))
    ps = PorterStemmer()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load and preprocess data
def load_data(directory):
    labels = []
    texts = []
    for label in os.listdir(directory):
        class_dir = os.path.join(directory, label)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    cleaned_text = preprocess_text(text)
                    texts.append(cleaned_text)
                    labels.append(label)
    return texts, labels

# Path to your dataset directory
dataset_path = r"C:\Users\alexx\OneDrive\Escriptori\uni\3rd year\Efficient-Recognition-of-Official-Documents\Tests Alex Sanchez\text_extracted" 
texts, labels = load_data(dataset_path)

# Create a bag of words model
vectorizer = CountVectorizer()
X_tfidf = vectorizer.fit_transform(texts)
y = labels

#Adjust weights for specific keywords.
keywords = ["calificaciones", "evaluación"]
adjustment_factor = 100
for keyword in keywords:
    if keyword in vectorizer.vocabulary_:
        keyword_index = vectorizer.vocabulary_[keyword]
        # The following lines adjust the column of the sparse matrix
        col = X_tfidf[:, keyword_index]
        # Multiply the non-zero elements of the column by the adjustment factor
        X_tfidf[:, keyword_index] = col.multiply(adjustment_factor)

# Stratified Split of the data to ensure each label appears in the test set
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.4, stratify=y, random_state=1)

# Train a classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}') # la accuracy del BOW es 0.6. Siii yujuuu que locurote, da pena voy a probar mas cosas.


# Visualization.
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Textos mal clasificados.
for true_label, pred_label, text in zip(y_test, y_pred, y_test):
    if true_label != pred_label:
        print(f"True: {true_label}, Pred: {pred_label}, Text: {text[:100]}...")

top_n=15

# Get the word frequencies
word_freq = X_tfidf.sum(axis=0)

# Convert to a dense array and flatten
word_freq = np.array(word_freq).flatten()

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Sort the features based on their frequency
sorted_indices = np.argsort(word_freq)[::-1]
sorted_word_freq = word_freq[sorted_indices]
sorted_feature_names = np.array(feature_names)[sorted_indices]

# Plot the top N words and their frequencies
top_n = 20  # Change this to visualize more or fewer words
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names[:top_n], sorted_word_freq[:top_n])
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.title('Top {} Words Frequency'.format(top_n))
plt.gca().invert_yaxis()
plt.show()

