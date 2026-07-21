from unittest.mock import Mock

import numpy as np

from app.config import Settings
from app.llm import provider


def test_sentence_transformer_embedding_provider(monkeypatch):
    model = Mock()
    model.encode.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
    constructor = Mock(return_value=model)
    monkeypatch.setattr(provider, "SentenceTransformer", constructor)

    embeddings = provider.SentenceTransformerEmbeddingProvider("multilingual-model")

    assert embeddings.embed([]) == []
    assert embeddings.embed(["سلام", "hello"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert embeddings.fingerprint == "sentence-transformers:multilingual-model"
    constructor.assert_called_once_with("multilingual-model")
    model.encode.assert_called_once_with(["سلام", "hello"], convert_to_numpy=True)


def test_embedding_factory_uses_configured_local_model(monkeypatch):
    constructed = Mock()
    monkeypatch.setattr(provider, "SentenceTransformerEmbeddingProvider", constructed)
    settings = Settings(_env_file=None, EMBEDDING_MODEL="custom-model")

    provider.create_embedding_provider(settings)

    constructed.assert_called_once_with("custom-model")
