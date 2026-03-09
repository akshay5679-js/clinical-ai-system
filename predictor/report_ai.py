import fitz
import ollama


def analyze_report(pdf_path):

    text = ""

    # extract text from pdf
    doc = fitz.open(pdf_path)

    for page in doc:
        text += page.get_text()

    prompt = f"""
You are a medical AI assistant.

Analyze the following medical report and provide:

1. Short summary
2. Important findings
3. Possible health risks

Report:
{text}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]