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
        "What is the recipient address?",
        "What products/services are listed on the invoice?"
    ]

    document_info = {}

    for question in questions:
        answer = nlp(pdf_image_url, question)[0]
        document_info[question] = answer["answer"]

    invoice_number = document_info["What is the invoice number?"]
    total_amount_str = document_info["What is the total invoice amount?"]
    amount_value_str, currency_str = total_amount_str.split(' ')  # Need to check if this works for all cases
    amount_value = float(amount_value_str.replace(',', '.'))  # Save as float
    recipient_address = document_info["What is the recipient address?"]
    invoice_items = document_info["What products/services are listed on the invoice?"]

    # Store the amount value and currency in a dictionary
    amount_info = {
        "value": amount_value,
        "currency": currency_str
    }

    return {
        "invoice_number": invoice_number,
        "total_amount": amount_info,
        "recipient_address": recipient_address,
        "invoice_items": invoice_items
    }
