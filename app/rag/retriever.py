def to_retrieved_items(query_result: dict) -> list[dict]:
    docs = query_result.get("documents", [[]])[0]
    metadatas = query_result.get("metadatas", [[]])[0]

    items: list[dict] = []
    for i, doc in enumerate(docs):
        md = metadatas[i] if i < len(metadatas) else {}
        items.append({"text": doc, "metadata": md})

    return items
