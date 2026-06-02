def build_prompt(question: str, contexts: list[str]) -> str:
    context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return (
        "You are a RAG assistant. Answer only based on the provided context. "
        "If the context is insufficient, say: I don't know based on the provided documents.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )
