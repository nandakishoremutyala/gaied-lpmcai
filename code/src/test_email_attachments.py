import os
import logging
import pdfplumber
import pypandoc
from email import policy
from email.parser import BytesParser
from docx import Document
from bs4 import BeautifulSoup  # To extract text from HTML emails

# Configure Logging
logging.basicConfig(filename="email_processing.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Directories
EMAIL_DIR = "test/"
ATTACHMENTS_DIR = "attachments/"

# Ensure the attachments directory exists
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)


def read_eml_file(file_path):
    """Reads an .eml file and extracts subject, body, and attachments."""
    try:
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        subject = msg["subject"] if msg["subject"] else "(No Subject)"
        body = extract_body(msg)
        attachments = extract_attachments(msg)

        logging.info(f"Processed email: {subject}")
        return {"subject": subject, "body": body, "attachments": attachments}

    except Exception as e:
        logging.error(f"Error reading EML file {file_path}: {e}")
        return None


def extract_body(msg):
    """Extracts text from email body, supporting both Plain Text and HTML."""
    try:
        body_text = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()

                # Extract plain text
                if content_type == "text/plain":
                    body_text += part.get_payload(decode=True).decode(errors="ignore") + "\n"

                # Extract text from HTML (fallback)
                elif content_type == "text/html":
                    html_content = part.get_payload(decode=True).decode(errors="ignore")
                    soup = BeautifulSoup(html_content, "html.parser")
                    body_text += soup.get_text(separator=" ") + "\n"

        else:
            body_text = msg.get_payload(decode=True).decode(errors="ignore")

        return body_text.strip()

    except Exception as e:
        logging.error(f"Error extracting email body: {e}")
        return "(Error extracting body)"


def extract_attachments(msg):
    """Extracts and processes attachments (PDF, DOCX, DOC, TXT)."""
    extracted_data = {}

    for part in msg.walk():
        if part.get_content_disposition() == "attachment":
            filename = part.get_filename()

            if not filename:
                filename = f"attachment_{len(extracted_data) + 1}"

            file_path = os.path.join(ATTACHMENTS_DIR, filename)

            # Save the attachment
            with open(file_path, "wb") as f:
                f.write(part.get_payload(decode=True))

            # Extract text from the attachment
            text = process_attachment(file_path)
            extracted_data[filename] = text

    return extracted_data


def process_attachment(file_path):
    """Reads and extracts text from PDF, DOCX, DOC, TXT files."""
    try:
        if file_path.endswith(".pdf"):
            return extract_text_from_pdf(file_path)
        elif file_path.endswith(".docx"):
            return extract_text_from_docx(file_path)
        elif file_path.endswith(".doc"):
            return extract_text_from_doc(file_path)
        elif file_path.endswith(".txt"):
            return extract_text_from_txt(file_path)
        else:
            logging.warning(f"Unsupported file type: {file_path}")
            return "(Unsupported file type)"
    except Exception as e:
        logging.error(f"Error processing attachment {file_path}: {e}")
        return "(Error processing attachment)"


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text if text else "(No readable text in PDF)"
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}")
        return "(Error reading PDF)"


def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text if text else "(No readable text in DOCX)"
    except Exception as e:
        logging.error(f"Error reading DOCX {docx_path}: {e}")
        return "(Error reading DOCX)"


def extract_text_from_doc(doc_path):
    """Extracts text from a DOC file using pypandoc."""
    try:
        return pypandoc.convert_file(doc_path, "plain")
    except Exception as e:
        logging.error(f"Error reading DOC {doc_path}: {e}")
        return "(Error reading DOC)"


def extract_text_from_txt(txt_path):
    """Extracts text from a TXT file."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Error reading TXT {txt_path}: {e}")
        return "(Error reading TXT)"


def process_all_emails(directory):
    """Processes all .eml files in the specified directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".eml"):
            file_path = os.path.join(directory, filename)
            email_data = read_eml_file(file_path)
            if email_data:
                logging.info(f"Extracted email data: {email_data}")
                print(email_data)  # Display extracted data in console


if __name__ == "__main__":
    process_all_emails(EMAIL_DIR)
