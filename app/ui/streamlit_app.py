import html

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="centered")
st.title("RAG Chat")

st.markdown(
    """
    <style>
      .rag-msg {
        direction: rtl;
        unicode-bidi: plaintext;
        text-align: right;
        line-height: 1.8;
        font-size: 1rem;
        word-break: break-word;
      }
      .rag-msg p {
        margin: 0 0 0.55rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_mixed_text(text: str):
    lines = text.splitlines() or [text]
    paragraphs = "".join(
        f'<p dir="auto">{html.escape(line) if line else "&nbsp;"}</p>' for line in lines
    )
    st.markdown(f'<div class="rag-msg">{paragraphs}</div>', unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_mixed_text(msg["content"])

user_input = st.chat_input("پیام خود را بنویس...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        render_mixed_text(user_input)

    with st.chat_message("assistant"):
        with st.spinner("در حال فکر کردن..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": user_input, "debug": False},
                    timeout=600,
                )
                if not resp.ok:
                    answer = f"خطا: {resp.text}"
                    sources = []
                else:
                    data = resp.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
            except requests.RequestException as exc:
                answer = f"خطا در اتصال به بک‌اند: {exc}"
                sources = []

        render_mixed_text(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- {s.get('file_name', '')} (chunk {s.get('chunk_index', 0)})")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_sources = sources
