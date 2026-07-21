from app.rag.prompting import build_messages, build_prompt, system_policy


def test_prompt_delimits_context_and_history():
    prompt = build_prompt("سؤال", ["متن منبع"], [("user", "سلام")])
    assert '<source id="S1">' in prompt
    assert "user: سلام" in prompt
    assert "When the customer writes in Persian" in prompt


def test_messages_preserve_native_roles_and_branding():
    messages = build_messages(
        "نسخه داره؟",
        ["نسخه فعلی ۲ است."],
        [("user", "محصول چیه؟"), ("assistant", "یک افزونه مرورگر است. [S1]")],
        assistant_name="یار",
        company_name="شرکت نمونه",
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "یار" in messages[0]["content"]
    assert "شرکت نمونه" in messages[0]["content"]
    assert '<source id="S1">' in messages[-1]["content"]


def test_policy_requires_natural_grounded_persian_answers():
    policy = system_policy("یار", "شرکت نمونه")

    assert "natural Iranian Persian" in policy
    assert "Ask at most one focused follow-up question" in policy
    assert "Never invent" in policy
    assert "numbered steps" in policy
    assert "[S1]" in policy
