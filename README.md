
# Resume Screening Dashboard

Streamlit app to screen and rank resumes against a job description.

## Features
- Upload multiple resumes (PDF/DOCX/TXT)
- Paste job description
- TF-IDF cosine similarity scoring
- Simple skills extraction

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

## Tests
```bash
pytest
```
