import os

import weaviate
from dotenv import load_dotenv

from src.schema import DOC_CLASS

load_dotenv()

client = weaviate.Client(
    url="http://localhost:8080",  # Replace with your endpoint
    additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},  # Replace with your inference API key
)


def get_batch_with_cursor(client, class_name, batch_size=20, class_properties=None, filter=None, cursor=None):
    query = client.query.get(class_name, class_properties).with_where(filter).with_limit(batch_size)

    if cursor is not None:
        return query.with_after(cursor).do()
    else:
        return query.do()


def get_all_docs(features, filter):
    docs = get_batch_with_cursor(
        client,
        DOC_CLASS,
        class_properties=features,
        filter=filter,
    )
    return docs
