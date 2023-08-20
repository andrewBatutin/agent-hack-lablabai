import os
from logging import getLogger

import weaviate
from dotenv import load_dotenv

from src.indexing.utils import clear_up_docs, import_data, set_up_batch
from src.schema import DOC_CLASS, TAX_LIMIT

load_dotenv()

logger = getLogger(__name__)


def setup_schema(client):
    # ===== add schema =====
    invoice_schema = {
        "class": DOC_CLASS,
        "description": "Object to store invoice information",
        "invertedIndexConfig": {
            "bm25": {"b": 0.75, "k1": 1.2},
            "cleanupIntervalSeconds": 60,
            "stopwords": {"additions": None, "preset": "en", "removals": None},
        },
        "moduleConfig": {
            "text2vec-openai": {"model": "ada", "modelVersion": "002", "type": "text", "vectorizeClassName": True},
            "generative-openai": {},
        },
        "vectorIndexType": "hnsw",
        "vectorizer": "text2vec-openai",
    }
    tax_limit_schema = {
        "class": TAX_LIMIT,
        "description": "Object to store tax limits",
        "invertedIndexConfig": {
            "bm25": {"b": 0.75, "k1": 1.2},
            "cleanupIntervalSeconds": 60,
            "stopwords": {"additions": None, "preset": "en", "removals": None},
        },
        "moduleConfig": {
            "text2vec-openai": {"model": "ada", "modelVersion": "002", "type": "text", "vectorizeClassName": True},
            "generative-openai": {},
        },
        "vectorIndexType": "hnsw",
        "vectorizer": "text2vec-openai",
    }

    client.schema.create_class(invoice_schema)
    client.schema.create_class(tax_limit_schema)


def run_indexing():
    w_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    client = weaviate.Client(
        w_url,
        auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WV_API_KEY"]),
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )
    setup_schema(client)
    set_up_batch(client)
    clear_up_docs(client)
    import_data(client)
    logger.info("Finished importing data.")


run_indexing()
