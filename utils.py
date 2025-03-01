import re

def preprocess_text(text: str):
    """簡單文本預處理"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text