# crawler.py

import os
import aiohttp
import asyncio
import hashlib
import logging
from urllib.parse import urljoin

from bs4 import BeautifulSoup


TIMEOUT = 10  # seconds for each request
# We skip inline <svg> if it has only a <use xlink:href="...">
def is_use_only_svg(svg_text: str) -> bool:
    lower = svg_text.lower()
    return ("<use" in lower) and ("xlink:href" in lower)


class AsyncSVGCrawler:
    """
    An asynchronous crawler that:
      - fetches HTML pages
      - extracts inline & external SVG references
      - deduplicates each new SVG by hashing it
      - saves each new SVG to disk *immediately*
    """
    def __init__(self, session: aiohttp.ClientSession, output_folder: str):
        self.session = session
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        # Keep track of known SVG hashes
        self._known_hashes = set()
        # We use a Lock to ensure we don't write the same hash simultaneously
        self._write_lock = asyncio.Lock()

    async def fetch_html(self, url: str) -> str:
        """
        Async fetch for an HTML page. Returns empty string on failure or non-HTML content.
        """
        try:
            async with self.session.get(url, timeout=TIMEOUT) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "").lower()
                if "html" not in ctype:
                    return ""
                return await resp.text()
        except Exception as e:
            logging.debug(f"Failed to fetch HTML from {url}: {e}")
            return ""

    async def fetch_svg(self, url: str) -> str:
        """
        Async fetch for .svg content. Returns cleaned text or empty on failure.
        """
        try:
            async with self.session.get(url, timeout=TIMEOUT) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "").lower()
                if ("svg" in ctype) or url.lower().endswith(".svg"):
                    raw = await resp.text()
                    # remove backslashes
                    return raw.replace("\\", "")
        except Exception as e:
            logging.debug(f"Failed to fetch SVG from {url}: {e}")
        return ""

    async def process_url(self, url: str):
        """
        1) Fetch HTML
        2) Extract inline + external SVG
        3) Save each new one
        """
        html = await self.fetch_html(url)
        if not html:
            return

        soup = BeautifulSoup(html, "html.parser")

        # 1) Inline <svg>
        inline_svgs = soup.find_all("svg")
        for svg_tag in inline_svgs:
            svg_str = str(svg_tag)
            if not is_use_only_svg(svg_str):
                cleaned = svg_str.replace("\\", "")
                await self.save_if_new(cleaned)

        # 2) External .svg references: <img>, <object>, <embed> with src=...
        for tag in soup.find_all(["img", "object", "embed"], src=True):
            src = tag["src"]
            if src.lower().endswith(".svg"):
                abs_url = urljoin(url, src)
                svg_text = await self.fetch_svg(abs_url)
                if svg_text:
                    await self.save_if_new(svg_text)

        # Also <object data="something.svg">
        for obj in soup.find_all("object", data=True):
            data_url = obj["data"]
            if data_url.lower().endswith(".svg"):
                abs_url = urljoin(url, data_url)
                svg_text = await self.fetch_svg(abs_url)
                if svg_text:
                    await self.save_if_new(svg_text)

    async def save_if_new(self, svg_text: str):
        """
        Check if we've already saved this SVG (by MD5 hash).
        If not, save it immediately to disk.
        """
        h = hashlib.md5(svg_text.encode("utf-8")).hexdigest()
        if h not in self._known_hashes:
            # Use a lock to avoid race conditions writing same file
            async with self._write_lock:
                if h in self._known_hashes:
                    return
                self._known_hashes.add(h)
                out_path = os.path.join(self.output_folder, f"{h}.svg")
                # Write out the file
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(svg_text)
