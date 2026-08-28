import os
from pypdf import PdfReader
from backend.config import DATA_DIR

company_knowledge_files =[
    "company_info.pdf",
    "payment_policy.pdf",
    "return_policy.pdf",
    "shipping_policy.pdf",
    "warranty_policy.pdf",
]

chunk_size = 800
chunk_overlap = 150

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text: str, source:str) -> list[dict]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source})
        start += chunk_size - chunk_overlap
    return chunks

def load_policy_chunks() -> list[dict]:
    all_chunks = []
    for file_name in company_knowledge_files:
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            text = extract_text_from_pdf(file_path)
            all_chunks.extend(chunk_text(text, source=file_name))
        else:
            print(f"Warning: {file_path} does not exist.")
    return all_chunks