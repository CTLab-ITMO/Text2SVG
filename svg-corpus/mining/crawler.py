# crawler.py

import os
import aiohttp
import asyncio
import hashlib
import logging
from urllib.parse import urljoin
import re
from bs4 import BeautifulSoup, SoupStrainer
import time
import json
from datetime import datetime

# Default timeout in seconds for each request
DEFAULT_TIMEOUT = 10

# Regular expressions for faster parsing
SVG_TAG_PATTERN = re.compile(r'<svg[^>]*>.*?</svg>', re.DOTALL | re.IGNORECASE)
SVG_USE_PATTERN = re.compile(r'<use\s+.*?href=', re.IGNORECASE)

# We skip inline <svg> if it has only a <use xlink:href="...">
def is_use_only_svg(svg_text: str) -> bool:
    return bool(SVG_USE_PATTERN.search(svg_text))

class AsyncSVGCrawler:
    """
    An asynchronous crawler that:
      - fetches HTML pages
      - extracts inline & external SVG references
      - deduplicates each new SVG by hashing it
      - saves each new SVG to disk *immediately*
    """
    def __init__(self, session: aiohttp.ClientSession, output_folder: str, 
                timeout=DEFAULT_TIMEOUT, max_retries=2):
        self.session = session
        self.output_folder = output_folder  # Use provided output folder
        self.timeout = timeout
        self.max_retries = max_retries
        
        os.makedirs(self.output_folder, exist_ok=True)
        # Keep track of known SVG hashes
        self._known_hashes = set()
        # We use a Lock to ensure we don't write the same hash simultaneously
        self._write_lock = asyncio.Lock()
        # Track stats
        self.stats = {
            "pages_fetched": 0,
            "svg_found": 0,
            "svg_saved": 0,
            "failed_fetches": 0,
            "retried_requests": 0,
            "last_log_time": time.time(),
            "last_svg_count": 0
        }

    async def log_progress(self):
        """
        Log progress every minute showing new unique SVGs found
        """
        current_time = time.time()
        if current_time - self.stats["last_log_time"] >= 60:  # Every minute
            new_svgs = len(self._known_hashes) - self.stats["last_svg_count"]
            logging.info(f"New unique SVGs found in last minute: {new_svgs} (Total unique SVGs: {len(self._known_hashes)})")
            self.stats["last_log_time"] = current_time
            self.stats["last_svg_count"] = len(self._known_hashes)

    async def fetch_with_retry(self, url: str, fetch_func):
        """
        Generic retry wrapper for fetch operations
        """
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    self.stats["retried_requests"] += 1
                    await asyncio.sleep(0.5 * attempt)  # Exponential backoff
                
                return await fetch_func(url)
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self.max_retries:
                    self.stats["failed_fetches"] += 1
                    logging.debug(f"All retries failed for {url}: {e}")
                    return ""
                logging.debug(f"Retry {attempt+1}/{self.max_retries} for {url}: {e}")
            
            except Exception as e:
                self.stats["failed_fetches"] += 1
                logging.debug(f"Unexpected error fetching {url}: {e}")
                return ""
        
        return ""

    async def fetch_html(self, url: str) -> str:
        """
        Async fetch for an HTML page. Returns empty string on failure or non-HTML content.
        """
        async def _fetch(url):
            async with self.session.get(url, timeout=self.timeout, 
                                       headers={'Accept': 'text/html'}) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "").lower()
                if "html" not in ctype:
                    return ""
                
                self.stats["pages_fetched"] += 1
                return await resp.text()
        
        return await self.fetch_with_retry(url, _fetch)

    async def fetch_svg(self, url: str) -> str:
        """
        Async fetch for .svg content. Returns cleaned text or empty on failure.
        """
        async def _fetch(url):
            async with self.session.get(url, timeout=self.timeout,
                                      headers={'Accept': 'image/svg+xml'}) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "").lower()
                if ("svg" in ctype) or url.lower().endswith(".svg"):
                    raw = await resp.text()
                    # remove backslashes
                    return raw.replace("\\", "")
            return ""
            
        return await self.fetch_with_retry(url, _fetch)

    async def process_url(self, url: str):
        """
        1) Fetch HTML
        2) Extract inline + external SVG
        3) Save each new one
        Returns stats about the processing of this URL
        """
        start_time = time.time()
        result = {
            "url": url,
            "success": False,
            "svg_found": 0,
            "svg_saved": 0,
            "time_taken": 0
        }
        
        html = await self.fetch_html(url)
        if not html:
            result["time_taken"] = time.time() - start_time
            return result

        try:
            # Use SoupStrainer to only parse the relevant tags for better performance
            parse_only = SoupStrainer(['svg', 'img', 'object', 'embed'])
            soup = BeautifulSoup(html, "html.parser", parse_only=parse_only)
            
            # Alternatively, use regex for even faster inline SVG extraction
            inline_svg_count = 0
            svg_matches = SVG_TAG_PATTERN.findall(html)
            for svg_str in svg_matches:
                if not is_use_only_svg(svg_str):
                    cleaned = svg_str.replace("\\", "")
                    if await self.save_if_new(cleaned, url):
                        inline_svg_count += 1
            
            result["svg_found"] += inline_svg_count
            result["svg_saved"] += inline_svg_count
            
            # 1) Inline <svg> using BeautifulSoup as backup
            if inline_svg_count == 0:
                inline_svgs = soup.find_all("svg")
                for svg_tag in inline_svgs:
                    svg_str = str(svg_tag)
                    if not is_use_only_svg(svg_str):
                        cleaned = svg_str.replace("\\", "")
                        if await self.save_if_new(cleaned, url):
                            result["svg_found"] += 1
                            result["svg_saved"] += 1

            # 2) External .svg references: <img>, <object>, <embed> with src=...
            for tag in soup.find_all(["img", "object", "embed"], src=True):
                src = tag.get("src", "")
                if src and src.lower().endswith(".svg"):
                    abs_url = urljoin(url, src)
                    svg_text = await self.fetch_svg(abs_url)
                    if svg_text:
                        result["svg_found"] += 1
                        if await self.save_if_new(svg_text, url):
                            result["svg_saved"] += 1

            # Also <object data="something.svg">
            for obj in soup.find_all("object", data=True):
                data_url = obj.get("data", "")
                if data_url and data_url.lower().endswith(".svg"):
                    abs_url = urljoin(url, data_url)
                    svg_text = await self.fetch_svg(abs_url)
                    if svg_text:
                        result["svg_found"] += 1
                        if await self.save_if_new(svg_text, url):
                            result["svg_saved"] += 1
            
            result["success"] = True
            
        except Exception as e:
            logging.warning(f"Error processing {url}: {e}")
        
        result["time_taken"] = time.time() - start_time
        return result

    async def save_if_new(self, svg_text: str, source_url: str = None):
        """
        Check if we've already saved this SVG (by MD5 hash).
        If not, save it immediately to disk in JSONL format with metadata.
        Returns True if saved, False if already existed
        """
        MAX_SVG_LENGTH = 8192 * 4  # 32KB limit

        if len(svg_text) > MAX_SVG_LENGTH:
            logging.debug(f"SVG too long ({len(svg_text)} chars), skipping.")
            return False

        h = hashlib.md5(svg_text.encode("utf-8")).hexdigest()
        
        # Quick check without locking
        if h in self._known_hashes:
            return False
            
        # Use a lock to avoid race conditions writing same file
        async with self._write_lock:
            if h in self._known_hashes:
                return False
                
            self._known_hashes.add(h)
            out_path = os.path.join(self.output_folder, f"{h[:2]}", f"{h}.jsonl")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            # Create metadata dictionary
            metadata = {
                'url': source_url,
                'timestamp': datetime.utcnow().isoformat(),
                'svg': svg_text
            }
            
            # Write out the file in JSONL format
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(metadata) + "\n")
            
            self.stats["svg_saved"] += 1
            # Log progress
            await self.log_progress()
            return True
