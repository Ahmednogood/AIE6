import os
import fitz  # PyMuPDF
from typing import List

class TextFileLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_documents(self) -> List[str]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_documents(self) -> List[str]:
        doc = fitz.open(self.file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return [text]

class CharacterTextSplitter:
    def __init__(self, chunk_size: int = 300):
        self.chunk_size = chunk_size

    def split_texts(self, documents: List[str]) -> List[str]:
        chunks = []
        for doc in documents:
            for i in range(0, len(doc), self.chunk_size):
                chunk = doc[i: i + self.chunk_size]
                if len(chunk) > 0:
                    chunks.append(chunk)
        return chunks
