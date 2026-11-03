import pandas as pd
import re
import nltk
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
nltk.download('punkt', quiet=True)

print("Loading dataset...")
df = pd.read_csv('resume_dataset.csv')

print("Cleaning data...")
def cleanResume(resumeText):
    resumeText = re.sub('http\\S+\\s*', ' ', resumeText)
    resumeText = re.sub('RT|cc', ' ', resumeText)
    resumeText = re.sub('#\\S+', '', resumeText)
    resumeText = re.sub('@\\S+', ' ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^a-zA-Z0-9\\s]', ' ', resumeText)
    resumeText = re.sub('\\s+', ' ', resumeText)
    return resumeText.lower()

df['Resume'] = df['Resume'].apply(cleanResume)
df = df.dropna()

print(f"Dataset shape: {df.shape}")
print("Categories:", df['Category'].value_counts().head())

X = df['Resume']
y = df['Category']
label_classes = sorted(y.unique())
y_encoded = pd.Categorical(y, categories=label_classes).codes

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print("Training TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training MultinomialNB Classifier...")
clf = MultinomialNB(alpha=0.5)
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_classes))

print("Saving models...")
joblib.dump(clf, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
np.save('label_classes.npy', label_classes)

print("Training completed! Models saved: model.joblib, vectorizer.joblib, label_classes.npy")
