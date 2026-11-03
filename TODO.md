 # Resume Screening Web App - TODO

## Current Status
- [x] Analyzed files (notebook, dataset, README)
- [x] Plan approved by user

## Implementation Steps (Approved Plan)

1. **Update Resume_Screening.ipynb / train_model.py** [x] 
   - Completed train_model.py with TF-IDF + MultinomialNB (run `python train_model.py` to generate models)
   - Notebook edits pending due to formatting issues

2. **Create requirements.txt** [x]

3. **Create app.py (Flask backend)** [x]
   - Full app created: model loading, text extraction (PDF/DOCX/TXT), / & /predict routes

4. **Create templates/** [x]
   - index.html: upload form created
   - results.html: prediction results created


5. **Test & Demo** [ ] 
   - pip install -r requirements.txt
   - python train_model.py
   - python app.py
   - Test upload → verify prediction
   - Demo localhost:5000

**Next Step:** Test & Demo (Step 5).
