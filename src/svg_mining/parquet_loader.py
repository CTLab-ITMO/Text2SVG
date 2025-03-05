# parquet_loader.py

import os
import requests
import logging
import pyarrow.parquet as pq

def download_parquet(parquet_url: str) -> str:
    """
    Downloads a Parquet file from `parquet_url` if it does not already exist.
    Returns the local filename of the downloaded file.
    """
    local_filename = os.path.basename(parquet_url)

    if os.path.exists(local_filename):
        logging.info(f"Parquet file already exists locally: {local_filename}")
        return local_filename

    logging.info(f"Downloading {parquet_url} -> {local_filename}")
    with requests.get(parquet_url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    logging.info(f"Download complete: {local_filename}")
    return local_filename


def iterate_parquet_rows(local_filename: str, column_name="url"):
    """
    Yields row-group chunks of unique URLs from a Parquet file.
    Each yield is a list of unique URLs (strings).
    We assume the Parquet has a column named `column_name` = "url".
    """
    pqfile = pq.ParquetFile(local_filename)
    num_row_groups = pqfile.metadata.num_row_groups

    logging.info(f"{local_filename} has {num_row_groups} row-groups, total {pqfile.metadata.num_rows} rows.")

    for rg_idx in range(num_row_groups):
        table = pqfile.read_row_group(rg_idx, columns=[column_name])
        df = table.to_pandas()
        # Drop missing or duplicated
        urls = df[column_name].dropna().unique().tolist()
        yield urls