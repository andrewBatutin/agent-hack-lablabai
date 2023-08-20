from src.taix.client import get_all_docs


def test_get_docs():
    docs = get_all_docs(
        features=["value", "country"], filter={"path": ["country"], "operator": "Equal", "valueText": "Germany"}
    )
    assert len(docs["data"]["Get"]["Invoice"]) == 2
