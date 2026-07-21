from __future__ import annotations

import ipaddress
import socket
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit

import httpx

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - exercised only when the optional parser is absent
    BeautifulSoup = None  # type: ignore[assignment,misc]


class CrawlError(ValueError):
    """An intentionally safe error for rejected or unusable crawl targets."""


@dataclass(frozen=True)
class CrawledPage:
    url: str
    title: str
    text: str


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise CrawlError("Only absolute HTTP(S) URLs may be crawled.")
    if parsed.username or parsed.password:
        raise CrawlError("Crawl URLs must not include credentials.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise CrawlError("The crawl URL contains an invalid port.") from exc
    expected_port = 80 if parsed.scheme.lower() == "http" else 443
    if port not in {None, expected_port}:
        raise CrawlError("Only the default port for the URL scheme may be crawled.")
    netloc = parsed.hostname.lower()
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


def origin(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def assert_public_host(hostname: str) -> None:
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise CrawlError("The crawl host could not be resolved.") from exc
    if not addresses:
        raise CrawlError("The crawl host could not be resolved.")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise CrawlError("Crawl targets must resolve to public IP addresses.")


class WebCrawler:
    """Bounded same-origin HTML crawler with pre-request DNS screening."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        max_response_bytes: int,
        allowed_origins: list[str],
        client_factory: Callable[..., httpx.Client] = httpx.Client,
        host_validator: Callable[[str], None] = assert_public_host,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.allowed_origins = set(allowed_origins)
        self.client_factory = client_factory
        self.host_validator = host_validator

    def crawl(self, start_url: str, *, max_pages: int, max_depth: int) -> list[CrawledPage]:
        if BeautifulSoup is None:
            raise CrawlError("HTML crawling is unavailable because its parser is not installed.")
        start_url = canonicalize_url(start_url)
        start_origin = origin(start_url)
        if self.allowed_origins and start_origin not in self.allowed_origins:
            raise CrawlError("This crawl origin is not permitted.")

        queue: deque[tuple[str, int]] = deque([(start_url, 0)])
        seen: set[str] = set()
        pages: list[CrawledPage] = []
        requests_made = 0
        with self.client_factory(timeout=self.timeout_seconds, follow_redirects=False) as client:
            while queue and requests_made < max_pages:
                current_url, depth = queue.popleft()
                if current_url in seen:
                    continue
                seen.add(current_url)
                parsed = urlsplit(current_url)
                self.host_validator(parsed.hostname or "")
                requests_made += 1
                response = self._get_bounded(client, current_url)
                if response.status_code != 200:
                    continue
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type:
                    continue

                soup = BeautifulSoup(response.content, "html.parser")
                for tag in soup(["script", "style", "noscript", "template", "nav", "header", "footer"]):
                    tag.decompose()
                title = soup.title.get_text(" ", strip=True) if soup.title else ""
                text_node = soup.find("main") or soup.find("article") or soup.body or soup
                text = text_node.get_text(" ", strip=True)
                if text:
                    pages.append(CrawledPage(current_url, title, text))

                if depth >= max_depth:
                    continue
                for link in soup.find_all("a", href=True):
                    try:
                        candidate = canonicalize_url(urldefrag(urljoin(current_url, str(link["href"])))[0])
                    except CrawlError:
                        continue
                    if origin(candidate) == start_origin and candidate not in seen:
                        queue.append((candidate, depth + 1))
        return pages

    def _get_bounded(self, client: httpx.Client, url: str) -> httpx.Response:
        with client.stream(
            "GET", url, headers={"User-Agent": "customer-support-rag-crawler/1.0"}
        ) as response:
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > self.max_response_bytes:
                raise CrawlError("A crawled page exceeded the response-size limit.")
            body = bytearray()
            for piece in response.iter_bytes():
                body.extend(piece)
                if len(body) > self.max_response_bytes:
                    raise CrawlError("A crawled page exceeded the response-size limit.")
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                content=bytes(body),
                request=response.request,
            )
