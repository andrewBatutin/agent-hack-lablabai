import os

import weaviate
from dotenv import load_dotenv

from src.indexing.wv import DOC_CLASS

load_dotenv()

client = weaviate.Client(
    url="http://localhost:8080",  # Replace with your endpoint
    additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},  # Replace with your inference API key
)


def get_batch_with_cursor(client, class_name, class_properties, batch_size=20, cursor=None):
    query = (
        client.query.get(class_name, class_properties)
        # Optionally retrieve the vector embedding by adding `vector` to the _additional fields
        # .with_additional(["id vector"])
        .with_limit(batch_size)
    )

    if cursor is not None:
        return query.with_after(cursor).do()
    else:
        return query.do()


def get_all_docs():
    docs = get_batch_with_cursor(client, DOC_CLASS, ["file_name", "img_path", "pdf_path"])
    return docs
