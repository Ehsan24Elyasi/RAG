from app.rag.retriever import to_retrieved_items


def test_to_retrieved_items_maps_documents_and_metadata():
    result = {"documents": [["x"]], "metadatas": [[{"file_name": "a.txt", "chunk_index": 0}]]}
    items = to_retrieved_items(result)
    assert len(items) == 1
    assert items[0]["text"] == "x"
    assert items[0]["metadata"]["file_name"] == "a.txt"
