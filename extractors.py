import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

def extract_text_from_url(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    text = " ".join(soup.stripped_strings)
    return url, text

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for p in reader.pages:
        text += p.extract_text() or ""
    return path, text

def extract_text_from_raw_text(title, text):
    return title, text
