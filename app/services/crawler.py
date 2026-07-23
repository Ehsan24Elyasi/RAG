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


@dataclass(frozen=True)
class _FetchedPage:
    url: str
    response: httpx.Response


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise CrawlError("Only absolute HTTP(S) URLs may be crawled.")
    if parsed.username or parsed.password:
        raise CrawlError("Crawl URLs must not include credentials.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise CrawlError("The crawl URL contains an invalid port.") from exc
    expected_port = 80 if scheme == "http" else 443
    if port not in {None, expected_port}:
        raise CrawlError("Only the default port for the URL scheme may be crawled.")
    hostname = parsed.hostname.lower()
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def origin(url: str) -> str:
    parsed = urlsplit(canonicalize_url(url))
    return f"{parsed.scheme}://{parsed.netloc}"


def assert_public_host(hostname: str) -> None:
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise CrawlError("The crawl host could not be resolved.") from exc
    if not addresses:
        raise CrawlError("The crawl host could not be resolved.")
    for address in addresses:
        if not ipaddress.ip_address(address).is_global:
            raise CrawlError("Crawl targets must resolve to public IP addresses.")


class WebCrawler:
    """Bounded same-origin HTML crawler with validated manual redirects."""

    redirect_statuses = {301, 302, 303, 307, 308}

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
        self.allowed_origins = {origin(item) for item in allowed_origins}
        self.client_factory = client_factory
        self.host_validator = host_validator

    def crawl(self, start_url: str, *, max_pages: int, max_depth: int) -> list[CrawledPage]:
        if BeautifulSoup is None:
            raise CrawlError("HTML crawling is unavailable because its parser is not installed.")
        start_url = canonicalize_url(start_url)
        if self.allowed_origins and origin(start_url) not in self.allowed_origins:
            raise CrawlError("This crawl origin is not permitted.")

        queue: deque[tuple[str, int]] = deque([(start_url, 0)])
        queued = {start_url}
        attempted: set[str] = set()
        final_seen: set[str] = set()
        pages: list[CrawledPage] = []
        effective_origin: str | None = None

        try:
            with self.client_factory(timeout=self.timeout_seconds, follow_redirects=False) as client:
                while queue and len(attempted) < max_pages:
                    current_url, depth = queue.popleft()
                    if current_url in attempted:
                        continue
                    attempted.add(current_url)
                    fetched = self._fetch_with_redirects(
                        client,
                        current_url,
                        effective_origin=effective_origin if effective_origin else None,
                        is_start=effective_origin is None,
                    )
                    final_url, response = fetched.url, fetched.response
                    if effective_origin is None:
                        effective_origin = origin(final_url)
                    if final_url in final_seen:
                        continue
                    final_seen.add(final_url)
                    if not 200 <= response.status_code < 300:
                        if current_url == start_url:
                            raise CrawlError(f"The crawl target returned HTTP {response.status_code}.")
                        continue
                    if not response.content:
                        if current_url == start_url:
                            raise CrawlError("The crawl target contained no usable page text.")
                        continue
                    if not self._is_html(response, allow_sniff=current_url == start_url):
                        if current_url == start_url:
                            raise CrawlError("The crawl target did not return an HTML page.")
                        continue

                    soup = BeautifulSoup(response.content, "html.parser")
                    links = list(soup.find_all("a", href=True))
                    title = soup.title.get_text(" ", strip=True) if soup.title else ""
                    for tag in soup(["script", "style", "noscript", "template"]):
                        tag.decompose()
                    text_node = soup.find("main") or soup.find("article") or soup.body or soup
                    for tag in text_node.find_all(["nav", "header", "footer"]):
                        tag.decompose()
                    text = text_node.get_text(" ", strip=True)
                    if not text:
                        if current_url == start_url:
                            raise CrawlError("The crawl target contained no usable page text.")
                        continue
                    pages.append(CrawledPage(final_url, title, text))

                    if depth >= max_depth:
                        continue
                    assert effective_origin is not None
                    for link in links:
                        try:
                            href = str(link.get("href", ""))
                            candidate = canonicalize_url(urldefrag(urljoin(final_url, href))[0])
                        except CrawlError:
                            continue
                        if origin(candidate) != effective_origin or candidate in queued:
                            continue
                        queued.add(candidate)
                        queue.append((candidate, depth + 1))
        except CrawlError:
            raise
        except httpx.TimeoutException as exc:
            raise CrawlError("The crawl target timed out.") from exc
        except httpx.HTTPError as exc:
            raise CrawlError("The crawl target could not be reached securely.") from exc

        if not pages:
            raise CrawlError("The crawl did not return any usable HTML pages.")
        return pages

    def _fetch_with_redirects(
        self,
        client: httpx.Client,
        url: str,
        *,
        effective_origin: str | None,
        is_start: bool,
        max_redirects: int = 5,
    ) -> _FetchedPage:
        current = canonicalize_url(url)
        visited: set[str] = set()
        for redirects in range(max_redirects + 1):
            if current in visited:
                raise CrawlError("The crawl target contains a redirect loop.")
            visited.add(current)
            current_origin = origin(current)
            if self.allowed_origins and current_origin not in self.allowed_origins:
                raise CrawlError("This crawl origin is not permitted.")
            if effective_origin and current_origin != effective_origin:
                raise CrawlError("A crawled page redirected outside the permitted origin.")
            parsed = urlsplit(current)
            self.host_validator(parsed.hostname or "")
            response = self._get_bounded(client, current)
            if response.status_code not in self.redirect_statuses:
                return _FetchedPage(current, response)
            location = response.headers.get("location")
            if not location:
                raise CrawlError("The crawl target returned an invalid redirect.")
            if redirects >= max_redirects:
                raise CrawlError("The crawl target exceeded the redirect limit.")
            target = canonicalize_url(urldefrag(urljoin(current, location))[0])
            target_origin = origin(target)
            if effective_origin and target_origin != effective_origin:
                raise CrawlError("A crawled page redirected outside the permitted origin.")
            if not is_start and target_origin != origin(url):
                raise CrawlError("A crawled page redirected outside the permitted origin.")
            current = target
        raise CrawlError("The crawl target exceeded the redirect limit.")

    @staticmethod
    def _is_html(response: httpx.Response, *, allow_sniff: bool) -> bool:
        media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type in {"text/html", "application/xhtml+xml"}:
            return True
        if media_type or not allow_sniff:
            return False
        prefix = response.content[:512].lstrip().lower()
        return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")

    def _get_bounded(self, client: httpx.Client, url: str) -> httpx.Response:
        with client.stream(
            "GET", url, headers={"User-Agent": "customer-support-rag-crawler/1.0"}
        ) as response:
            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > self.max_response_bytes:
                        raise CrawlError("A crawled page exceeded the response-size limit.")
                except ValueError:
                    pass
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
