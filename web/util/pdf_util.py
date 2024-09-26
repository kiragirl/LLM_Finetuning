from PyPDF2 import PdfReader


class PDFUtil:

    # 读取 PDF 文件中的文本
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        text = ""
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text

    # 将文本分割成片段
    @staticmethod
    def split_into_chunks(text, chunk_size=512):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size
        return chunks
