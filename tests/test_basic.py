
from app import clean_text, extract_skills

def test_clean_text_removes_symbols():
    text = "Hello!! Python 3.9"
    cleaned = clean_text(text)
    assert "python" in cleaned
    assert "hello" in cleaned

def test_extract_skills():
    text = "I know Python and AWS."
    skills = extract_skills(text, ["python","aws","sql"])
    assert set(skills) == {"python","aws"}
