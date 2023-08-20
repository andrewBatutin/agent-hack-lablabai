import os
from logging import getLogger

import weaviate

from src.indexing.utils import clear_up_docs, import_data, set_up_batch

logger = getLogger(__name__)

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
if not WEAVIATE_URL:
    WEAVIATE_URL = "http://localhost:8080"


def run_indexing():
    client = weaviate.Client(WEAVIATE_URL, additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]})
    set_up_batch(client)
    clear_up_docs(client)
    import_data(client)
    logger.info("Finished importing data.")


# run_indexing()
