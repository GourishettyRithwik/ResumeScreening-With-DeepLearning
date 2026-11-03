import pandas as pd
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score

CATEGORY_SKILLS = {
    'Data Science': ['python', 'r', 'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'sql', 'statistics', 'data analysis', 'big data', 'hadoop', 'spark'],
    'HR': ['recruitment', 'talent acquisition', 'employee relations', 'performance management', 'training', 'payroll', 'compensation', 'labor law', 'succession planning', 'hr analytics'],
    'Business Analyst': ['requirements gathering', 'data analysis', 'sql', 'excel', 'tableau', 'power bi', 'agile', 'jira', 'user stories', 'process improvement', 'stakeholder management'],
    'Java Developer': ['java', 'spring boot', 'hibernate', 'microservices', 'rest api', 'maven', 'jenkins', 'docker', 'kubernetes', 'sql', 'nosql'],
    'Testing': ['selenium', 'junit', 'testng', 'cypress', 'jmeter', 'postman', 'automation', 'manual testing', 'agile', 'defect tracking'],
    'DevOps Engineer': ['docker', 'kubernetes', 'jenkins', 'ansible', 'terraform', 'aws', 'azure', 'ci cd', 'monitoring', 'prometheus', 'grafana'],
    'Python Developer': ['python', 'django', 'flask', 'fastapi', 'pandas', 'numpy', 'sqlalchemy', 'celery', 'docker', 'aws'],
    'Blockchain': ['solidity', 'ethereum', 'hyperledger', 'smart contracts', 'web3', 'truffle', 'ganache', 'dapp', 'ipfs'],
    'C++': ['c++', 'stl', 'boost', 'multithreading', 'opencv', 'qt', 'linux', 'performance optimization'],
    'Hadoop': ['hadoop', 'spark', 'hive', 'hbase', 'sqoop', 'flume', 'kafka', 'mapreduce', 'yarn', 'pig']
}

print("Loading dataset...")
df = pd.read_csv('resume_dataset.csv', encoding='utf-8')

def clean_resume(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

print("Preprocessing resumes...")
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

le = LabelEncoder()
y = le.fit_transform(df['Category'])
X = df['cleaned_resume'].values

print("Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=2500, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Training Deep Learning MLP (Multi-Layer Perceptron) ---")
# Deep Feedforward Neural Network:
# Layer 1: Input (TF-IDF features: 2500)
# Layer 2: Dense Hidden Layer (128 neurons, ReLU activation)
# Layer 3: Dense Hidden Layer (64 neurons, ReLU activation)
# Output Layer: Softmax across all 25 resume categories
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, y_pred_mlp)
print(f"Deep Learning MLP Test Accuracy: {mlp_acc:.2%}")

print("\n--- Training Hybrid Ensemble with Deep Learning (MLP + Naive Bayes + KNN) ---")
knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
nb = OneVsRestClassifier(MultinomialNB(alpha=1.0))

ensemble = VotingClassifier(
    estimators=[
        ('mlp_neural_network', mlp),
        ('naive_bayes', nb),
        ('knn', knn)
    ],
    voting='soft'
)
ensemble.fit(X_train, y_train)

y_pred_ensemble = ensemble.predict(X_test)
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
print(f"Hybrid Ensemble Test Accuracy: {ensemble_acc:.2%}")

print("\nClassification Report (Hybrid Deep Learning Ensemble):")
print(classification_report(y_test, y_pred_ensemble, target_names=le.classes_))

# Fit final ensemble on full dataset for maximum production coverage
print("Fitting final model on full dataset...")
ensemble.fit(X_tfidf, y)

print("Saving models to root directory...")
joblib.dump(ensemble, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(le, 'label_encoder.joblib')
joblib.dump(CATEGORY_SKILLS, 'category_skills.pkl')

print("Deploying updated models to python-ml-service/...")
joblib.dump(ensemble, 'python-ml-service/model.joblib')
joblib.dump(vectorizer, 'python-ml-service/vectorizer.joblib')
joblib.dump(le, 'python-ml-service/label_encoder.joblib')
joblib.dump(CATEGORY_SKILLS, 'python-ml-service/category_skills.pkl')

print("\n[SUCCESS] Deep Learning MLP Neural Network integrated & saved successfully!")
