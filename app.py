from flask import Flask, render_template, request, jsonify
import joblib
import os
import re
import traceback
import numpy as np
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import pytesseract
from PIL import Image
import io

# --- Configure Tesseract Path (for Windows Users) ---
# This helps the app find Tesseract if it's not in your system PATH.
TESSERACT_PATHS = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    r'C:\Users\\' + os.environ.get('USERNAME', 'Default') + r'\AppData\Local\Tesseract-OCR\tesseract.exe'
]

for path in TESSERACT_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break

# Try to handle different PyPDF2 versions (3.0.0+ uses PdfReader)
try:
    from PyPDF2 import PdfReader
    PDF_READER_CLASS = PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfFileReader
        PDF_READER_CLASS = PdfFileReader
    except ImportError:
        PDF_READER_CLASS = None

app = Flask(__name__)

# --- Load ML models and artifacts ONCE at startup ---
print("--- Loading ML Engine ---")
try:
    clf = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    le = joblib.load('label_encoder.joblib')
    category_skills = joblib.load('category_skills.pkl')
    print("[OK] All models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Error loading models: {e}")
    clf, vectorizer, le, category_skills = None, None, None, None

def clean_text(text):
    if not text: return ""
    text = re.sub('http\\S+\\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\\S+', '', text)
    text = re.sub('@\\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\\x00-\\x7f]', r' ', text)
    text = re.sub('\\s+', ' ', text)
    return text.lower().strip()

def extract_text_from_file(file):
    filename = file.filename.lower()
    text = ""
    try:
        if filename.endswith('.pdf'):
            if PDF_READER_CLASS is None:
                return "[Error: PyPDF2 not installed]"
            
            reader = PDF_READER_CLASS(file)
            pages = reader.pages if hasattr(reader, 'pages') else reader.getNumPages()
            
            page_count = len(pages) if hasattr(pages, '__len__') else pages
            max_pages = min(page_count, 50)
            
            for i in range(max_pages):
                page = pages[i] if hasattr(pages, '__getitem__') else reader.getPage(i)
                text += page.extract_text() or ""
                
        elif filename.endswith('.docx'):
            doc = Document(file)
            text = '\n'.join([para.text for para in doc.paragraphs])
        elif filename.endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Use PIL to open the image
                image = Image.open(file)
                text = pytesseract.image_to_string(image)
                if not text.strip():
                    text = "[Error: No text could be extracted from the image. Quality might be low.]"
            except Exception as e:
                if "tesseract" in str(e).lower() or "not found" in str(e).lower():
                    text = "[Error: Tesseract OCR not found. Please install it and add to PATH.]"
                else:
                    text = f"[OCR Error: {str(e)}]"
        else:
            text = file.read().decode('utf-8', errors='ignore')
            
        print(f"[OK] Extracted {len(text)} characters from {filename}")
        return text
    except Exception as e:
        print(f"[ERROR] Extraction Error in {filename}: {e}")
        return f"[Extraction Error: {str(e)}]"

def get_skills(text, category):
    if not category_skills or category not in category_skills:
        return [], []
    
    found_skills = []
    missing_skills = []
    
    target_skills = category_skills[category]
    text_lower = text.lower()
    
    for skill in target_skills:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            found_skills.append(skill)
        else:
            missing_skills.append(skill)
            
    return found_skills, missing_skills

def calculate_jd_match(resume_text, jd_text):
    if not jd_text.strip() or not vectorizer:
        return 0
    
    try:
        texts = [clean_text(resume_text), clean_text(jd_text)]
        vectors = vectorizer.transform(texts)
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return round(float(similarity) * 100, 2)
    except Exception as e:
        print(f"[ERROR] JD Match Error: {e}")
        return 0

def get_skill_based_scores(text):
    if not category_skills: return {}
    scores = {}
    text_lower = text.lower()
    for cat, skills in category_skills.items():
        found = [s for s in skills if re.search(r'\b' + re.escape(s.lower()) + r'\b', text_lower)]
        if found:
            # Score as percentage of matched skills for that domain
            # We add a slight boost for each skill found
            scores[cat] = (len(found) / len(skills)) * 100
    return scores

def get_job_suggestions(category, found_skills=[]):
    # Comprehensive sub-roles dictionary for variety across different resumes
    sub_roles = {
        'Data Science': ['Data Scientist', 'Machine Learning Engineer', 'Data Analyst', 'AI Researcher'],
        'HR': ['Human Resource Manager', 'Talent Acquisition', 'HR Specialist', 'Recruiter'],
        'Java Developer': ['Java Backend Engineer', 'Software Developer (Java)', 'Full Stack Java Developer'],
        'DevOps Engineer': ['DevOps Architect', 'Cloud Infrastructure Engineer', 'SRE', 'Platform Engineer'],
        'Python Developer': ['Python Backend Developer', 'Django Specialist', 'Automation Engineer'],
        'Blockchain': ['Blockchain Developer', 'Smart Contract Engineer', 'Solidity Specialist'],
        'Business Analyst': ['Senior Business Analyst', 'Systems Analyst', 'Product Analyst'],
        'Web Designing': ['Frontend Developer', 'UI/UX Designer', 'Web Architect', 'React Developer'],
        'Testing': ['QA Engineer', 'Automation Specialist', 'Manual Tester', 'SDET'],
        'Sales': ['Sales Manager', 'Business Development', 'Account Executive', 'Sales Consultant'],
        'Network Security Engineer': ['Cybersecurity Analyst', 'Network Administrator', 'Security Architect'],
        'Database': ['DBA', 'Database Developer', 'Data Architect', 'SQL specialist'],
        'Operations Manager': ['Operations Lead', 'Project Manager', 'Supply Chain Manager', 'Process Analyst']
    }
    
    # Base titles
    titles = sub_roles.get(category, [category, f"Junior {category}", f"Senior {category}"]).copy()
    
    # Dynamic Variety Enhancement based on extracted skills
    if any(s in found_skills for s in ['React', 'Vue', 'Next.js', 'Frontend']):
        titles.insert(0, "Modern Frontend Engineer")
    if any(s in found_skills for s in ['AWS', 'Azure', 'GCP', 'Cloud']):
        titles.insert(0, "Cloud Solutions Architect")
    if any(s in found_skills for s in ['Python', 'Pandas', 'NumPy', 'TensorFlow']):
        titles.insert(0, "Data/ML Operations Engineer")

    # URL Encode for searching
    safe_query = urllib.parse.quote(category)
    
    return {
        'titles': list(dict.fromkeys(titles))[:5], # Unique titles, limit to 5
        'links': {
            'linkedin': f"https://www.linkedin.com/jobs/search/?keywords={safe_query}",
            'indeed': f"https://www.indeed.com/jobs?q={safe_query}",
            'glassdoor': f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={safe_query}"
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['file']
        jd_text = request.form.get('jd', '')
        
        if file.filename == '':
            return "No file selected", 400
        
        resume_text = extract_text_from_file(file)
        if not resume_text or resume_text.startswith("["):
            return render_template('index.html', error=f"Could not read file: {resume_text}")
        
        if clf is None or vectorizer is None:
            return "Server Error: Models not initialized.", 500
            
        cleaned_resume = clean_text(resume_text)
        features = vectorizer.transform([cleaned_resume])
        
        # Multi-Job Match Logic (Hybrid ML + Skill Density)
        # This prevents "same list for all resumes" by injecting actual skill extraction into results
        probs = clf.predict_proba(features)[0]
        all_categories = le.classes_
        
        # 1. Base ML Matches
        ml_matches = {all_categories[i]: round(float(probs[i]) * 100, 2) for i in range(len(all_categories))}
        
        # 2. Skill Density Scoring
        skill_scores = get_skill_based_scores(resume_text)
        
        # 3. Hybrid Score Calculation
        # We blend the results so that Alternative domains are based on PERSONAL skills too.
        hybrid_matches = []
        for cat in all_categories:
            ml_score = ml_matches.get(cat, 0)
            skill_score = skill_scores.get(cat, 0)
            
            # Combine: 70% pure prediction, 30% actual skills found
            # This ensures that if you have DevOps skills, DevOps shows up, even if the model is biased.
            combined = (ml_score * 0.7) + (skill_score * 0.3)
            
            if combined >= 0.5: # Show even low-chance alternatives for variety
                hybrid_matches.append({'category': cat, 'score': round(combined, 2)})
        
        # Sort by hybrid score
        hybrid_matches = sorted(hybrid_matches, key=lambda x: x['score'], reverse=True)
        top_matches = hybrid_matches[:5]
        
        primary_match = top_matches[0] if top_matches else {'category': 'None', 'score': 0}
        
        # Supplemental Analysis
        found_skills, missing_skills = get_skills(resume_text, primary_match['category'])
        jd_match_score = calculate_jd_match(resume_text, jd_text) if jd_text else None
        
        # --- NEW: Live Job Suggestions (Now with Variety Injection) ---
        job_suggestions = get_job_suggestions(primary_match['category'], found_skills)
        
        print(f"[OK] Prediction: {primary_match['category']} | Suggestions: Linked to Top Portals")
        
        return render_template('results.html', 
                               category=primary_match['category'], 
                               confidence=primary_match['score'],
                               top_matches=top_matches,
                               found_skills=found_skills,
                               missing_skills=missing_skills,
                               jd_match_score=jd_match_score,
                               job_suggestions=job_suggestions,
                               resume_preview=resume_text[:1000] + "...")

    except Exception as e:
        print(f"[CRITICAL ERROR]: {e}")
        print(traceback.format_exc())
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
