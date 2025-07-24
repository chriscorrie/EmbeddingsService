import os
from PyPDF2 import PdfReader
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation

def extract_text_from_file(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.xlsx'):
        return extract_text_from_excel(file_path)
    elif file_path.endswith('.pptx'):
        return extract_text_from_pptx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return '\n'.join(page.extract_text() for page in reader.pages)

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

def extract_text_from_excel(file_path):
    workbook = load_workbook(file_path, data_only=True)
    text = []
    for sheet in workbook.sheetnames:
        worksheet = workbook[sheet]
        for row in worksheet.iter_rows(values_only=True):
            text.append(' '.join(map(str, row)))
    return '\n'.join(text)

def extract_text_from_pptx(file_path):
    presentation = Presentation(file_path)
    text = []
    for slide in presentation.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                text.append(shape.text)
    return '\n'.join(text)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

if __name__ == '__main__':
    # Example usage
    sample_file = '/mnt/HomerShare/FBO Attachments/sample.pdf'
    print(extract_text_from_file(sample_file))
