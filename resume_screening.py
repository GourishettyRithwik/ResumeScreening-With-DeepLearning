import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

import re
import string
import os

try:
    import nltk
    from nltk.corpus import stopwords
    from wordcloud import WordCloud
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK or WordCloud not available. Skipping wordcloud.")
    NLTK_AVAILABLE = False

# Load dataset
print("Loading dataset...")
resumeDataSet = pd.read_csv('resume_dataset.csv', encoding='utf-8')
print(f"Dataset shape: {resumeDataSet.shape}")
print("\nFirst few rows:")
print(resumeDataSet.head())

print("\nDistinct categories:")
print(resumeDataSet['Category'].unique())

print("\nCategory counts:")
print(resumeDataSet['Category'].value_counts())

# Plot category distribution
plt.figure(figsize=(15, 8))
sns.countplot(y="Category", data=resumeDataSet, order=resumeDataSet['Category'].value_counts().index)
plt.title('Category Distribution')
plt.savefig('category_dist.png', dpi=300, bbox_inches='tight')
print("\nCategory distribution plot saved as 'category_dist.png'")

# Pie chart
targetCounts = resumeDataSet['Category'].value_counts()
targetLabels = resumeDataSet['Category'].unique()
plt.figure(figsize=(12, 12))
colors = plt.cm.Set3(np.linspace(0, 1, len(targetLabels)))
plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.title('Category Distribution - Pie Chart')
plt.savefig('category_pie.png', dpi=300, bbox_inches='tight')
print("Pie chart saved as 'category_pie.png'")

# Clean resumes
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)
    resumeText = re.sub('RT|cc', ' ', resumeText)
    resumeText = re.sub('#\S+', '', resumeText)
    resumeText = re.sub('@\S+', ' ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText.lower()

print("\nCleaning resumes...")
resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))
print("Cleaning completed.")

# Wordcloud (use first 1000 for performance)
print("\nGenerating wordcloud...")
sample_size = min(1000, len(resumeDataSet))
cleanedSentences = ""
totalWords = []
oneSetOfStopWords = set(stopwords.words('english') + ['``', "''"]) if NLTK_AVAILABLE else set()

for i in range(sample_size):
    cleanedText = resumeDataSet['Resume'].iloc[i]
    cleanedText = cleanResume(cleanedText)
    cleanedSentences += cleanedText
    if NLTK_AVAILABLE:
        requiredWords = nltk.word_tokenize(cleanedText)
        for word in requiredWords:
            if word not in oneSetOfStopWords and word not in string.punctuation:
                totalWords.append(word)

if NLTK_AVAILABLE:
    wordfreqdist = nltk.FreqDist(totalWords)
    mostcommon = wordfreqdist.most_common(50)
    print("Top 50 words:", mostcommon)

    wc = WordCloud(width=800, height=400, background_color='white').generate(cleanedSentences)
    plt.figure(figsize=(15, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Resume Word Cloud')
    plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight')
    print("Wordcloud saved as 'wordcloud.png'")
else:
    print("Skipping wordcloud due to missing NLTK/WordCloud.")

# Encode categories
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])

# Prepare data
requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

# TF-IDF
print("\nCreating TF-IDF features...")
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500
)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)
print("TF-IDF features shape:", WordFeatures.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train model
print("\nTraining KNN model...")
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

# Predictions
prediction = clf.predict(X_test)

# Results
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print(f'\nAccuracy of KNeighbors Classifier on training set: {train_accuracy:.2f}')
print(f'Accuracy of KNeighbors Classifier on test set: {test_accuracy:.2f}')
print("\nClassification report:\n", metrics.classification_report(y_test, prediction, target_names=le.classes_))

print("\nScript completed successfully!")
print("Generated files: category_dist.png, category_pie.png, wordcloud.png (if NLTK available)")
