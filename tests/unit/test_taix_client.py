from src.taix.client import get_all_docs


def test_get_docs():
    docs = get_all_docs()
    assert len(docs["data"]["Get"]["Invoice"]) == 10
