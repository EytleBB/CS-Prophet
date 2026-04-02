from __future__ import annotations

import random
import time
from typing import Any

import cloudscraper


class HLTVScraper:
    """
    Rate-limited HTTP client for HLTV pages.

    Usage:
        cfg = {"min_delay": 2, "max_delay": 5, "max_retries": 3}
        s = HLTVScraper(cfg)
        html = s.get("https://www.hltv.org/results")
        s.download_file("https://www.hltv.org/download/demo/123", "/tmp/demo.rar")
    """

    BASE = "https://www.hltv.org"

    def __init__(self, rate_limit_cfg: dict[str, Any]) -> None:
        self._min_delay = rate_limit_cfg.get("min_delay", 2)
        self._max_delay = rate_limit_cfg.get("max_delay", 5)
        self._max_retries = rate_limit_cfg.get("max_retries", 3)
        self._session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
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
            resp = self._session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (429, 503):
                wait = 2 ** (attempt + 2)
                print(f"  [{resp.status_code}] backing off {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Failed to fetch {url} after {self._max_retries} retries")

    def download_file(self, url: str, dest_path: str) -> None:
        """
        Stream-download a file (demo archive) to dest_path.
        Retries on transient errors.
        """
        if not url.startswith("http"):
            url = self.BASE + url
        for attempt in range(self._max_retries):
            self._sleep()
            resp = self._session.get(url, stream=True, timeout=60)
            if resp.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return
            if resp.status_code in (429, 503):
                wait = 2 ** (attempt + 2)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Failed to download {url} after {self._max_retries} retries")
