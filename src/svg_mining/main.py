# main.py

import asyncio
import argparse
import logging
import os
import aiohttp
from tqdm import tqdm
from parquet_loader import download_parquet, iterate_parquet_rows
from crawler import AsyncSVGCrawler

async def process_urls_in_chunk(urls, crawler: AsyncSVGCrawler, sem: asyncio.Semaphore, pbar):
    """
    Process a batch of URLs concurrently, respecting the concurrency limit (sem),
    updating the progress bar after each URL is done.
    """
    tasks = []
    for url in urls:
        tasks.append(asyncio.create_task(process_single_url(url, crawler, sem, pbar)))
    await asyncio.gather(*tasks)

async def process_single_url(url: str, crawler: AsyncSVGCrawler, sem: asyncio.Semaphore, pbar):
    """
    Acquire the semaphore, run crawler.process_url(url), release, then update progress bar by 1.
    """
    async with sem:
        await crawler.process_url(url)
    pbar.update(1)

async def run(parquet_url: str, max_concurrency: int):
    """
    1) Download the parquet file if necessary
    2) Determine output folder from the filename
    3) Create an aiohttp session + crawler
    4) Read row-groups from the parquet in chunks
    5) For each chunk (list of URLs), schedule tasks
    6) Show a TQDM progress bar across *total row count*
    """
    # 1) Download
    local_file = download_parquet(parquet_url)
    base_name = os.path.splitext(local_file)[0]
    out_folder = "falcon_urls_" + base_name
    os.makedirs(out_folder, exist_ok=True)

    # 2) Count total row groups / rows
    import pyarrow.parquet as pq
    pqfile = pq.ParquetFile(local_file)
    total_rows = pqfile.metadata.num_rows

    # 3) Create session + crawler
    connector = aiohttp.TCPConnector(limit=max_concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        crawler = AsyncSVGCrawler(session, out_folder)
        sem = asyncio.Semaphore(max_concurrency)

        # 4) Setup progress bar
        with tqdm(total=total_rows, desc="Processing URLs") as pbar:
            # 5) Iterate row-groups
            for urls in iterate_parquet_rows(local_file, column_name="url"):
                # Each chunk is a list of unique URLs
                await process_urls_in_chunk(urls, crawler, sem, pbar)

    logging.info("Done! All row-groups processed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-url", required=True, help="URL to the Parquet file containing a 'url' column.")
    parser.add_argument("--max-concurrency", type=int, default=1_000, help="Max concurrent fetches (default=20).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    asyncio.run(run(args.parquet_url, args.max_concurrency))

if __name__ == "__main__":
    main()