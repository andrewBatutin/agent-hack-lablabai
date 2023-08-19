from price_parser import Price
from transformers import pipeline

# Initialize LayoutLM document-question-answering pipe
nlp = pipeline(
    "document-question-answering",
    model="magorshunov/layoutlm-invoices",  # https://huggingface.co/magorshunov/layoutlm-invoices
)


def extract_invoice_info(pdf_image_url):
    # List of questions for extracting predefined info
    questions = [
        "What is the invoice number?",
        "What is the total invoice amount?",
        "What is the recipient's address?",
        "What is the recipient country?",
        "What products/services are listed on the invoice?",
    ]

    document_info = {}

    for question in questions:
        answer = nlp(pdf_image_url, question)[0]
        document_info[question] = answer["answer"]

    invoice_number = document_info["What is the invoice number?"]
    total_amount_str = document_info["What is the total invoice amount?"]
    price = Price.fromstring(total_amount_str)
    amount_value, currency_str = float(price.amount), price.currency
    recipient_address = document_info["What is the recipient's address?"]
    invoice_items = document_info["What products/services are listed on the invoice?"]
    country = document_info["What is the recipient country?"]

    return {
        "invoice_number": invoice_number,
        "value": amount_value,
        "currency": currency_str,
        "recipient_address": recipient_address,
        "invoice_items": invoice_items,
        "country": country,
    }
