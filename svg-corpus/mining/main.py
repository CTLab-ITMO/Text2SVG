# main.py

import asyncio
import argparse
import logging
import os
import aiohttp
from tqdm import tqdm
from datasets import load_dataset
from crawler import AsyncSVGCrawler
import random
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Maximum number of connection errors before skipping a URL
MAX_RETRIES = 2

# Define reasonable timeouts
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(
    total=30,       # Total timeout for the whole request
    connect=5,      # Connection timeout
    sock_read=15    # Socket read timeout
)

async def process_urls_in_chunk(urls, crawler: AsyncSVGCrawler, sem: asyncio.Semaphore, pbar, debug=False):
    """
    Process a batch of URLs concurrently, respecting the concurrency limit (sem),
    updating the progress bar after each URL is done.
    """
    if debug:
        logging.debug(f"Processing chunk of {len(urls)} URLs")
    
    # Create tasks for all URLs immediately
    tasks = []
    for url in urls:
        task = asyncio.create_task(process_single_url(url, crawler, sem, pbar, debug))
        tasks.append(task)
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

async def process_single_url(url: str, crawler: AsyncSVGCrawler, sem: asyncio.Semaphore, pbar, debug=False):
    """
    Acquire the semaphore, run crawler.process_url(url), release, then update progress bar by 1.
    """
    if debug:
        logging.debug(f"Processing URL: {url}")
    
    async with sem:
        try:
            result = await crawler.process_url(url)
            if debug and result:
                logging.debug(f"Result for {url}: {result}")
        except Exception as e:
            if debug:
                logging.debug(f"Error processing {url}: {str(e)}")
    
    pbar.update(1)

def iterate_dataset_urls(dataset_name, column_name="url", start_offset=0, batch_size=1000, max_urls=None, debug=False):
    """
    Stream URLs from a Hugging Face dataset, yielding them in batches.
    """
    if debug:
        logging.debug(f"Loading dataset: {dataset_name}, column: {column_name}, offset: {start_offset}")
        if max_urls is not None:
            logging.debug(f"Maximum URLs to process: {max_urls}")
    
    dataset = load_dataset(dataset_name, streaming=True).shuffle() # don't change shuffling seed
    
    if debug:
        logging.debug(f"Dataset splits: {list(dataset.keys())}")
    
    if 'train' in dataset:
        stream = dataset['train']
        if debug:
            logging.debug("Using 'train' split")
    else:
        # Use the first split if 'train' doesn't exist
        first_split = list(dataset.keys())[0]
        stream = dataset[first_split]
        if debug:
            logging.debug(f"Using '{first_split}' split")
    
    # Skip to the start_offset
    stream = stream.skip(start_offset)
    if debug:
        logging.debug(f"Skipped {start_offset} items")
    
    # Create batches and yield URLs - use smaller internal batches for more continuous processing
    current_batch = []
    batch_count = 0
    url_count = 0
    invalid_count = 0
    seen_urls = set()  # Track URLs we've already seen to avoid duplicates
    
    # Define a smaller internal batch size to yield more frequently
    internal_batch_size = min(batch_size, 10000)  # Never use internal batches larger than 10k
    
    for i, item in enumerate(stream):
        # Check if we've reached the URL limit
        if max_urls is not None and url_count >= max_urls:
            if debug:
                logging.debug(f"Reached URL limit of {max_urls}, stopping")
            break
            
        if column_name in item:
            url = item[column_name]
            if url and isinstance(url, str) and url not in seen_urls:
                seen_urls.add(url)
                current_batch.append(url)
                url_count += 1
            else:
                invalid_count += 1
                if debug and invalid_count <= 5:  # Limit to prevent log flooding
                    logging.debug(f"Invalid URL found in item {i+start_offset}: {url}")
                
            # Yield smaller internal batches for more continuous processing
            if len(current_batch) >= internal_batch_size:
                if debug:
                    batch_count += 1
                    logging.debug(f"Yielding batch {batch_count} with {len(current_batch)} URLs")
                yield current_batch
                current_batch = []
                
                # Check if we've reached the limit after yielding
                if max_urls is not None and url_count >= max_urls:
                    if debug:
                        logging.debug(f"Reached URL limit of {max_urls}, stopping")
                    break
        elif debug and i < 5:  # Only show first few missing columns to avoid log flooding
            logging.debug(f"Item at position {i+start_offset} missing column '{column_name}': {item.keys()}")
    
    # Yield any remaining URLs
    if current_batch:
        if debug:
            batch_count += 1
            logging.debug(f"Yielding final batch {batch_count} with {len(current_batch)} URLs")
        yield current_batch
    
    if debug:
        logging.debug(f"Total URLs processed: {url_count}, Invalid URLs: {invalid_count}")

async def run(dataset_name: str, column_name: str, max_concurrency: int, start_offset: int, 
              debug: bool, batch_size: int, timeout: int, output_dir: str, max_urls: int | None = None):
    """
    1) Create output folder
    2) Create an aiohttp session + crawler
    3) Stream URLs from the HF dataset
    4) For each chunk (list of URLs), schedule tasks
    5) Show a TQDM progress bar
    """
    # Create output folder
    out_folder = output_dir
    os.makedirs(out_folder, exist_ok=True)
    
    if debug:
        logging.debug(f"Created output folder: {out_folder}")
        logging.debug(f"Max concurrency: {max_concurrency}")

    # Create session + crawler with optimized settings for speed
    connector = aiohttp.TCPConnector(
        limit=max_concurrency * 4,  # Increase the connection limit
        ttl_dns_cache=300,  # Cache DNS results for 5 minutes
        ssl=False,          # Disable SSL verification for speed
        force_close=False,  # Allow keep-alive connections
        use_dns_cache=True, # Enable DNS caching
        keepalive_timeout=10 # Keep connections alive but with shorter timeout
    )
    
    # Very aggressive timeout settings for speed
    client_timeout = aiohttp.ClientTimeout(
        total=timeout,
        connect=min(3, timeout//2),  # Faster connect timeout
        sock_read=max(2, timeout-2)  # Faster socket read timeout
    )
    
    if debug:
        logging.debug(f"Created TCP connector with optimized settings")
        logging.debug(f"Timeout settings: {client_timeout}")
    
    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        crawler = AsyncSVGCrawler(session, out_folder)
        sem = asyncio.Semaphore(max_concurrency)
        
        if debug:
            logging.debug("Created HTTP session and crawler")

        with tqdm(desc="Processing URLs") as pbar:
            url_queue = asyncio.Queue(maxsize=10)
            
            consumers = []
            num_consumers = 8
            
            # Define consumer function
            async def url_consumer(consumer_id):
                while True:
                    try:
                        urls = await asyncio.wait_for(url_queue.get(), timeout=5)  # Add timeout to prevent hanging
                        if urls is None:  # End signal
                            break
                        await process_urls_in_chunk(urls, crawler, sem, pbar, debug)
                        url_queue.task_done()
                    except asyncio.TimeoutError:
                        # Check if producer is done and queue is empty
                        if producer.done() and url_queue.empty():
                            break
                        continue  # Otherwise keep waiting
                    except Exception as e:
                        logging.error(f"Consumer {consumer_id} error: {e}")
                if debug:
                    logging.debug(f"Consumer {consumer_id} finished")
            
            # Start URL producer
            async def url_producer():
                try:
                    count = 0
                    for urls in iterate_dataset_urls(dataset_name, column_name, start_offset, 
                                                   batch_size=batch_size, max_urls=max_urls, debug=debug):
                        await url_queue.put(urls)
                        count += 1
                    # Send end signals to all consumers
                    for _ in range(num_consumers):
                        await url_queue.put(None)
                    if debug:
                        logging.debug(f"Producer finished, processed {count} batches")
                except Exception as e:
                    logging.error(f"Producer error: {e}")
                    # Make sure consumers don't hang
                    for _ in range(num_consumers):
                        await url_queue.put(None)
            
            # Start producer and consumers
            producer = asyncio.create_task(url_producer())
            for i in range(num_consumers):
                consumers.append(asyncio.create_task(url_consumer(i)))
            
            # Wait for all tasks to complete with timeout
            try:
                await asyncio.wait_for(producer, timeout=None)  # No timeout for producer
                await asyncio.wait_for(asyncio.gather(*consumers), timeout=60)  # 60 sec timeout for consumers
            except asyncio.TimeoutError:
                logging.warning("Some consumers timed out, but we'll continue")
            except Exception as e:
                logging.error(f"Error in main processing: {e}")

    logging.info("Done! All URLs processed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name (e.g., 'username/dataset')")
    parser.add_argument("--column", default="url", help="Column name containing URLs (default='url')")
    parser.add_argument("--max-concurrency", type=int, default=500, 
                      help="Max concurrent fetches (default=500)")
    parser.add_argument("--start-offset", type=int, default=0,
                      help="Skip this many URLs before processing")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging")
    parser.add_argument("--batch-size", type=int, default=50000, 
                      help="Batch size for URL processing (default=50000)")
    parser.add_argument("--timeout", type=int, default=1,
                      help="Timeout in seconds for HTTP requests (default=1)")
    parser.add_argument("--output-dir", required=True,
                      help="Directory where SVG files will be saved")
    parser.add_argument("--max-urls", type=int, default=None,
                      help="Maximum number of URLs to process (default=None, no limit)")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    if args.debug:
        logging.debug("Debug mode enabled")
        logging.debug(f"Arguments: {args}")
    
    asyncio.get_event_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=24)
    )

    asyncio.run(run(
        args.dataset, 
        args.column, 
        args.max_concurrency, 
        args.start_offset, 
        args.debug,
        args.batch_size,
        args.timeout,
        args.output_dir,  # Pass output directory
        args.max_urls     # Pass max URLs limit
    ))

if __name__ == "__main__":
    main()
