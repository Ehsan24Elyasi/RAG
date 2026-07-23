import httpx
import pytest

from app.services.crawler import CrawlError, WebCrawler, canonicalize_url


def test_canonicalize_url_accepts_standard_web_urls():
    assert canonicalize_url("https://Example.com/help#top") == "https://example.com/help"


def test_canonicalize_url_rejects_credentials_and_custom_ports():
    with pytest.raises(CrawlError):
        canonicalize_url("https://user:pass@example.com/help")
    with pytest.raises(CrawlError):
        canonicalize_url("http://example.com:8080/help")
    with pytest.raises(CrawlError):
        canonicalize_url("https://example.com:80/help")


def _crawler(handler):
    transport = httpx.MockTransport(handler)
    return WebCrawler(
        timeout_seconds=2,
        max_response_bytes=100_000,
        allowed_origins=[],
        client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs),
        host_validator=lambda _: None,
    )


def test_crawler_follows_validated_redirect_and_discovers_navigation_links():
    def handler(request):
        if str(request.url) == "http://example.com/":
            return httpx.Response(301, headers={"location": "https://www.example.com/help"})
        if str(request.url) == "https://www.example.com/help":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="<html><head><title>Help</title></head><body><nav><a href='/faq'>FAQ</a></nav><main>Support home</main></body></html>",
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/xhtml+xml"},
            text="<html><body><main>Frequently asked questions</main></body></html>",
        )

    pages = _crawler(handler).crawl("http://example.com", max_pages=2, max_depth=1)
    assert [page.url for page in pages] == [
        "https://www.example.com/help",
        "https://www.example.com/faq",
    ]
    assert pages[0].text == "Support home"


def test_crawler_reports_non_html_start_page():
    crawler = _crawler(
        lambda request: httpx.Response(200, headers={"content-type": "application/json"}, json={"ok": True})
    )
    with pytest.raises(CrawlError, match="did not return an HTML page"):
        crawler.crawl("https://example.com", max_pages=1, max_depth=0)


def test_crawler_rejects_private_redirect_before_request():
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    crawler = WebCrawler(
        timeout_seconds=2,
        max_response_bytes=100_000,
        allowed_origins=[],
        client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs),
        host_validator=lambda host: (
            (_ for _ in ()).throw(CrawlError("private")) if host == "127.0.0.1" else None
        ),
    )
    with pytest.raises(CrawlError, match="private"):
        crawler.crawl("https://example.com", max_pages=1, max_depth=0)
    assert requested == ["https://example.com/"]
