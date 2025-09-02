import re
import os
import nltk
import pdfplumber
import docx2txt
import pandas as pd
import streamlit as st
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

DEFAULT_SKILLS = ["python", "java", "sql", "aws", "machine learning", "deep learning", "nlp"]

# ---------------------- Utility Functions ---------------------- #
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

def extract_skills(text: str, skills_list) -> list:
    found = []
    lower_text = text.lower()
    for s in skills_list:
        if s.lower() in lower_text:
            found.append(s)
    return sorted(set(found))

def score_resume(job_desc_clean: str, resume_clean: str) -> float:
    vect = TfidfVectorizer()
    X = vect.fit_transform([job_desc_clean, resume_clean])
    return float(cosine_similarity(X[0:1], X[1:2])[0][0])

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}", text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,5}[-.\s]?\d{4}", text)
    return match.group(0) if match else None

def extract_github(text):
    match = re.search(r"(https?:\/\/)?(www\.)?github\.com\/[A-Za-z0-9_-]+", text)
    return match.group(0) if match else None

def extract_education(text):
    degrees = ["bachelor", "master", "phd", "b.tech", "m.tech", "mba", "msc", "bsc"]
    found = [deg for deg in degrees if deg in text.lower()]
    return ", ".join(set(found)) if found else None

def extract_experience_years(text):
    match = re.findall(r"(\d+)\+?\s+(?:years|yrs|year)", text.lower())
    if match:
        return max([int(x) for x in match])
    return None

def extract_name(text):
    # Take first line as candidate name (simple heuristic)
    lines = text.strip().split("\n")
    if lines:
        first_line = lines[0].strip()
        if len(first_line.split()) <= 4:  # usually name is short
            return first_line
    return None

# ---------------------- Main App ---------------------- #
def main():
    st.title("📄 Resume Screening Dashboard")

    # 1. Job description input
    jd = st.text_area("Paste Job Description")
    uploaded_files = st.file_uploader(
        "Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )

    if st.button("Run Screening"):
        if not jd or not uploaded_files:
            st.error("Please provide inputs.")
            return

        jd_clean = clean_text(jd)
        rows = []

        # Make shortlisted folder
        shortlisted_dir = "shortlisted_resumes"
        os.makedirs(shortlisted_dir, exist_ok=True)

        # 2. Process each uploaded resume
        for uf in uploaded_files:
            file_bytes = uf.read()
            uf.seek(0)  # reset pointer

            # --- Extract text depending on file type ---
            if uf.name.endswith(".pdf"):
                try:
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                except Exception as e:
                    st.warning(f"⚠️ Could not read {uf.name} as PDF ({e})")
                    continue

            elif uf.name.endswith(".docx"):
                try:
                    text = docx2txt.process(io.BytesIO(file_bytes))
                except Exception as e:
                    st.warning(f"⚠️ Could not read {uf.name} as DOCX ({e})")
                    continue

            else:  # TXT
                text = file_bytes.decode("utf-8", errors="ignore")

            if not text.strip():
                st.warning(f"⚠️ No text extracted from {uf.name}")
                continue

            # --- Extract details ---
            candidate_name = extract_name(text)
            email = extract_email(text)
            phone = extract_phone(text)
            github = extract_github(text)
            education = extract_education(text)
            experience = extract_experience_years(text)
            skills = extract_skills(text, DEFAULT_SKILLS)
            match_score = score_resume(jd_clean, clean_text(text)) * 100

            rows.append({
                "Name": candidate_name,
                "Email": email,
                "Phone": phone,
                "GitHub": github,
                "Education": education,
                "Experience (yrs)": experience,
                "Skills": ", ".join(skills),
                "Match_Score": round(match_score, 2),
                "File": uf.name
            })

            # Save shortlisted resumes
            if match_score >= 75:
                with open(os.path.join(shortlisted_dir, uf.name), "wb") as f:
                    f.write(file_bytes)

        # 3. Show results
        if rows:
            df = pd.DataFrame(rows)

            st.subheader("📊 All Candidates")
            st.dataframe(df)

            shortlisted = df[df["Match_Score"] >= 75]

            st.subheader("✅ Shortlisted Candidates (≥ 75%)")
            if not shortlisted.empty:
                st.dataframe(shortlisted)
            else:
                st.info("No candidates scored above 75%.")

            # Download buttons
            st.download_button(
                "⬇️ Download All Candidates (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name="all_candidates.csv",
                mime="text/csv"
            )

            if not shortlisted.empty:
                st.download_button(
                    "⬇️ Download Shortlisted (CSV)",
                    shortlisted.to_csv(index=False).encode("utf-8"),
                    file_name="shortlisted_candidates.csv",
                    mime="text/csv"
                )

# ---------------------- Run ---------------------- #
if __name__ == "__main__":
    main()

