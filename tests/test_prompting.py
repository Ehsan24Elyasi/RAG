from app.rag.prompting import build_prompt


def test_prompt_delimits_context_and_history():
    prompt = build_prompt("سؤال", ["متن منبع"], [("user", "سلام")])
    assert '<source id="S1">' in prompt
    assert "user: سلام" in prompt
    assert "Reply in the language used by the customer" in prompt
