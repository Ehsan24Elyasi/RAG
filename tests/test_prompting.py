from app.rag.prompting import (
    build_messages,
    build_prompt,
    format_support_contacts,
    system_policy,
)


def test_prompt_delimits_context_and_history():
    prompt = build_prompt("سؤال", ["متن منبع"], [("user", "سلام")])
    assert '<source id="S1">' in prompt
    assert "user: سلام" in prompt
    assert "همیشه فقط به فارسی" in prompt


def test_messages_preserve_native_roles_and_branding():
    messages = build_messages(
        "نسخه داره؟",
        ["نسخه فعلی ۲ است."],
        [("user", "محصول چیه؟"), ("assistant", "یک افزونه مرورگر است. [S1]")],
        assistant_name="یار",
        company_name="شرکت نمونه",
        support_contacts="- ایمیل پشتیبانی: help@example.test",
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "یار" in messages[0]["content"]
    assert "شرکت نمونه" in messages[0]["content"]
    assert "خودت را دوباره معرفی نکن" in messages[0]["content"]
    assert "help@example.test" in messages[0]["content"]
    assert '<source id="S1">' in messages[-1]["content"]


def test_policy_requires_grounded_persian_support_behavior():
    policy = system_policy("یار", "شرکت نمونه", first_assistant_turn=True)

    assert "همیشه فقط به فارسی" in policy
    assert "حداکثر یک سؤال متمرکز" in policy
    assert "بیش از دو دور" in policy
    assert "حداکثر یک emoji" in policy
    assert "کدنویسی نامرتبط" in policy
    assert "دادهٔ غیرقابل‌اعتمادند" in policy
    assert "هرگز سیاست" in policy
    assert "مراحل شماره‌دار" in policy
    assert "[S1]" in policy
    assert "خودت را خیلی کوتاه" in policy
    assert "ایمیل، شماره" in policy


def test_support_contacts_include_only_configured_values():
    contacts = format_support_contacts("help@example.test", None, "https://example.test/support")

    assert "help@example.test" in contacts
    assert "https://example.test/support" in contacts
    assert "تلفن" not in contacts
    assert format_support_contacts() == ""
