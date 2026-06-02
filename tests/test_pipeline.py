from app.rag.generator import generate_answer


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return "ok"


def test_generate_answer_uses_provider():
    answer = generate_answer("q", [{"text": "ctx", "metadata": {}}], FakeLLM())
    assert answer == "ok"
