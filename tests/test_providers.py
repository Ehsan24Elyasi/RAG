from unittest.mock import Mock

import numpy as np

from app.config import Settings
from app.llm import provider


def test_sentence_transformer_embedding_provider_normalizes_without_download(monkeypatch):
    model = Mock()
    model.encode.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
    constructor = Mock(return_value=model)
    monkeypatch.setattr(provider, "SentenceTransformer", constructor)

    embeddings = provider.SentenceTransformerEmbeddingProvider("multilingual-model")

    assert embeddings.embed([]) == []
    assert embeddings.embed(["سلام", "hello"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert embeddings.fingerprint == "sentence-transformers:multilingual-model:normalized-v1"
    constructor.assert_called_once_with("multilingual-model")
    model.encode.assert_called_once_with(["سلام", "hello"], convert_to_numpy=True, normalize_embeddings=True)


def test_openai_provider_preserves_native_roles(monkeypatch):
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="answer"), finish_reason="stop")
    ]
    client.chat.completions.create.return_value.model = "model"
    client.chat.completions.create.return_value.usage = None
    monkeypatch.setattr(provider, "OpenAI", Mock(return_value=client))
    chat = provider.OpenAICompatibleChatProvider("key", "model", "https://example.test/v1")

    result = chat.generate(
        [{"role": "system", "content": "policy"}, {"role": "user", "content": "سلام"}]
    )

    assert result.text == "answer"
    assert result.model == "model"
    assert client.chat.completions.create.call_args.kwargs["messages"] == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "سلام"},
    ]


def test_embedding_factory_uses_configured_local_model(monkeypatch):
    constructed = Mock()
    monkeypatch.setattr(provider, "SentenceTransformerEmbeddingProvider", constructed)
    settings = Settings(_env_file=None, EMBEDDING_MODEL="custom-model")

    provider.create_embedding_provider(settings)

    constructed.assert_called_once_with("custom-model")
