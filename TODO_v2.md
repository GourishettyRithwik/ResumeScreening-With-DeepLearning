# Resume Screening - Accuracy Improvement TODO (JD Matching + External Knowledge)

## Current Status
- [x] Previous TODO.md updates (upload, UI, basic prediction)
- [x] Plan approved for accuracy improvements

## Implementation Steps

### 1. **Create models.py** [x DONE]
   - Pre-train ensemble model (KNN + NB)
   - External skills dictionary (24 categories → keywords)
   - Save: model.joblib, vectorizer.joblib, le.joblib

### 2. **Update requirements.txt** [x DONE]
   - Add `joblib`
   - Optional: `sentence-transformers` for embeddings

### 3. **Update resume_app.py** [PENDING]
   - Add JD input textarea
   - Load pre-trained model/vectorizer/LE
   - New scoring: cosine_sim(resume, JD) * category_conf + skills_bonus
   - Skill gap recommendations
   - Update UI: JD section, real match gauge

### 4. **Train & Test** [PENDING]
   - `python models.py`
   - `pip install -r requirements.txt`
   - `streamlit run resume_app.py`
   - Test JD matching accuracy

**Goal**: Improve prediction from generic category → job-specific match score using external JD + skills knowledge.
