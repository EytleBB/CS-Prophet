from __future__ import annotations

import os
import random
import time
from typing import Any

import functools

from curl_cffi import requests as cffi_requests

# Force unbuffered print in non-TTY mode
print = functools.partial(print, flush=True)  # type: ignore[assignment]


class HLTVScraper:
    """
    Rate-limited HTTP client for HLTV pages.
    Uses curl_cffi with Chrome impersonation to bypass Cloudflare.

    Usage:
        cfg = {"min_delay": 2, "max_delay": 5, "max_retries": 3}
        s = HLTVScraper(cfg)
        html = s.get("https://www.hltv.org/results")
        s.download_file("https://www.hltv.org/download/demo/123", "/tmp/demo.rar")
    """

    BASE = "https://www.hltv.org"

    def __init__(self, rate_limit_cfg: dict[str, Any], proxy: str | None = None) -> None:
        self._min_delay = rate_limit_cfg.get("min_delay", 2)
        self._max_delay = rate_limit_cfg.get("max_delay", 5)
        self._max_retries = rate_limit_cfg.get("max_retries", 3)
        # Proxy priority: explicit arg > env var
        self._proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("ALL_PROXY")
        if self._proxy:
            print(f"[proxy] using {self._proxy}")
        self._session = cffi_requests.Session(
            impersonate="chrome120",
            proxies={"http": self._proxy, "https": self._proxy} if self._proxy else None,
        )

    def _sleep(self) -> None:
        time.sleep(random.uniform(self._min_delay, self._max_delay))

    def get(self, url: str) -> str:
        """
        Fetch URL and return response text.
        Retries on 429/503 with exponential backoff.
        Raises RuntimeError after max_retries exhausted.
        """
        if not url.startswith("http"):
            url = self.BASE + url
        for attempt in range(self._max_retries):
            self._sleep()
            headers = {"Referer": self.BASE + "/results"}
            resp = self._session.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (403, 429, 503):
                wait = 2 ** (attempt + 2)
                print(f"  [{resp.status_code}] backing off {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Failed to fetch {url} after {self._max_retries} retries")

    def _resolve_cdn_url(self, url: str, referer: str | None = None) -> str:
        """
        Follow HLTV's demo redirect chain and return the final R2 CDN URL.
        Uses allow_redirects=False to only read Location headers, never the body.
        """
        hdrs = {
            "Referer": referer or self.BASE,
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
        }
        current_url = url
        for _ in range(10):  # max redirect hops
            self._sleep()
            resp = self._session.get(
                current_url, headers=hdrs, allow_redirects=False, timeout=15
            )
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location", "")
                if not location:
                    raise RuntimeError("Redirect response missing Location header")
                if not location.startswith("http"):
                    location = self.BASE + location
                if "r2-demos.hltv.org" in location:
                    return location
                current_url = location
                continue
            if resp.status_code in (429, 503):
                time.sleep(2 ** 3)
                continue
            raise RuntimeError(
                f"Unexpected status {resp.status_code} resolving CDN URL from {current_url}"
            )
        raise RuntimeError("Too many redirects resolving CDN URL")

    def download_file(self, url: str, dest_path: str, referer: str | None = None) -> None:
        """
        Download a demo archive to dest_path.
        Follows HLTV's redirect to R2 CDN in a single streaming request so that
        Cloudflare session context carries through the redirect chain.
        """
        if not url.startswith("http"):
            url = self.BASE + url

        for attempt in range(self._max_retries):
            self._sleep()
            hdrs = {
                "Referer": referer or self.BASE,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            try:
                resp = self._session.get(
                    url, headers=hdrs, allow_redirects=True, stream=True, timeout=1800
                )
            except Exception as e:
                print(f"    [retry] connection error (attempt {attempt+1}): {e}")
                time.sleep(2 ** (attempt + 2))
                continue
            if resp.status_code in (429, 503):
                time.sleep(2 ** (attempt + 2))
                continue
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                time.sleep(2 ** (attempt + 2))
                continue

            expected_size = int(resp.headers.get("Content-Length", 0))
            print(f"    [cdn] {resp.url} ({expected_size // 1024 // 1024} MB)")
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

            actual_size = os.path.getsize(dest_path)
            if expected_size > 0 and actual_size < expected_size:
                os.remove(dest_path)
                print(f"    [retry] incomplete: {actual_size}/{expected_size} bytes")
                time.sleep(2 ** (attempt + 2))
                continue

            with open(dest_path, "rb") as f:
                magic = f.read(4)
            if not (magic[:2] == b"PK" or magic[:4] == b"Rar!"):
                os.remove(dest_path)
                time.sleep(2 ** (attempt + 2))
                continue
            return

        raise RuntimeError(f"Failed to download demo after {self._max_retries} retries")
