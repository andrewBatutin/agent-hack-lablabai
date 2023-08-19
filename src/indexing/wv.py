import base64
import os
import re
from logging import getLogger

import weaviate

logger = getLogger(__name__)

DOC_PATH = "./data/invoices/"

DOC_CLASS = "Invoice"

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
if not WEAVIATE_URL:
    WEAVIATE_URL = "http://localhost:8080"


def set_up_batch():
    """
    Prepare batching configuration to speed up deleting and importing data.
    """
    client.batch.configure(
        batch_size=100,
        dynamic=True,
        timeout_retries=3,
        callback=None,
    )


def clear_up_docs():
    """
    Remove all objects from the DOC_CLASS collection.
    This is useful, if we want to rerun the import with different pictures.
    """
    try:
        with client.batch as batch:
            batch.delete_objects(
                class_name=DOC_CLASS,
                # same where operator as in the GraphQL API
                where={"operator": "NotEqual", "path": ["file_name"], "valueString": "x"},
                output="verbose",
            )
    except:
        logger.error("No objects to delete.")


def pdf_to_base64(file):
    file_bytes = base64.b64encode(file.read())
    base_64 = file_bytes.decode("ascii")
    return base_64


def import_data():
    """
    Process all pdf in [base64_images] folder and add import them into Dogs collection
    """

    with client.batch as batch:
        # Iterate over all .b64 files in the base64_images folder
        for encoded_file_path in os.listdir(DOC_PATH):
            with open(DOC_PATH + encoded_file_path, "rb") as file:
                base64_encoding = pdf_to_base64(file)

            # The properties from our schema
            data_properties = {
                "file_name": encoded_file_path,
                "pdf": base64_encoding,
                "filepath": file.name,
                # invoice amount
                # recipient address or country
                # date
                # invoice item/s
                # img file path
            }

            batch.add_data_object(data_properties, DOC_CLASS)


def run_indexing():
    global client
    client = weaviate.Client(WEAVIATE_URL)
    set_up_batch()
    clear_up_docs()
    import_data()
    logger.info("Finished importing data.")
