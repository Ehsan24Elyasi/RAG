from app.rag.prompting import build_prompt


def generate_answer(question: str, retrieved_items: list[dict], llm_provider) -> str:
    contexts = [item["text"] for item in retrieved_items]
    prompt = build_prompt(question, contexts)
    return llm_provider.generate(prompt)
