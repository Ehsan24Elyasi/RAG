import pytest

from app.services.crawler import CrawlError, canonicalize_url


def test_canonicalize_url_accepts_standard_web_urls():
    assert canonicalize_url("https://Example.com/help#top") == "https://example.com/help"


def test_canonicalize_url_rejects_credentials_and_custom_ports():
    with pytest.raises(CrawlError):
        canonicalize_url("https://user:pass@example.com/help")
    with pytest.raises(CrawlError):
        canonicalize_url("http://example.com:8080/help")
    with pytest.raises(CrawlError):
        canonicalize_url("https://example.com:80/help")
