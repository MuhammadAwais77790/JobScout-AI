from io import BytesIO

from docx import Document
from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from all pages of a PDF CV.
    """

    pdf_file = BytesIO(file_bytes)
    reader = PdfReader(pdf_file)

    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract text from paragraphs of a Word DOCX CV.
    """

    docx_file = BytesIO(file_bytes)
    document = Document(docx_file)

    text_parts = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text.strip())

    return "\n".join(text_parts).strip()


def extract_cv_text(file_bytes: bytes, filename: str) -> str:
    """
    Detect the CV file type and extract its text.
    """

    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)

    if filename_lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)

    raise ValueError(
        "Unsupported file type. Please upload a PDF or DOCX file."
    )