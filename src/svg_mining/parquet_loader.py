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


def iterate_parquet_rows(local_file, column_name="url", start_offset=0):
    import pyarrow.parquet as pq
    pqfile = pq.ParquetFile(local_file)
    logging.info(f"{local_file} has {pqfile.metadata.num_row_groups} row-groups, "
                 f"total {pqfile.metadata.num_rows} rows.")

    row_count = 0
    for rg_idx in range(pqfile.metadata.num_row_groups):
        table = pqfile.read_row_group(rg_idx, columns=[column_name])
        df = table.to_pandas()
        unique_urls = df[column_name].dropna().unique().tolist()

        # If the entire group is before our offset, skip it
        if row_count + len(unique_urls) <= start_offset:
            row_count += len(unique_urls)
            continue

        # If we're partially in this group, slice it
        if row_count < start_offset:
            skip = start_offset - row_count
            unique_urls = unique_urls[skip:]
            row_count = start_offset
        else:
            # Otherwise, we've passed the offset
            pass

        yield unique_urls
        row_count += len(unique_urls)

