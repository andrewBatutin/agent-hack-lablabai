FROM python:3.9-slim

ENV LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

COPY src/ /apps/src/
COPY requirements.txt /apps
COPY main.py /apps

WORKDIR /apps

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    poppler \
    poppler-utils \
    tesseract-ocr python3-pil  \
    tesseract-ocr-eng tesseract-ocr-script-latn \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONPATH .
EXPOSE 8000
CMD ["streamlit", "run", "main.py", "--server.port", "8000"]
