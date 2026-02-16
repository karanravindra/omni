import bz2
import os

import requests
from tqdm import tqdm

CHUNK_SIZE = 1024 * 1024  # 1 MB


def download_with_progress(url: str, output_path: str):
    """
    Download a file with a tqdm progress bar.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", 0))

        with (
            open(output_path, "wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {os.path.basename(output_path)}",
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def decompress_bz2_with_progress(bz2_path: str, output_path: str):
    """
    Decompress a .bz2 file with tqdm progress bar.
    """
    total_size = os.path.getsize(bz2_path)

    with (
        open(bz2_path, "rb") as compressed_file,
        open(output_path, "wb") as decompressed_file,
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Decompressing {os.path.basename(bz2_path)}",
        ) as pbar,
    ):
        decompressor = bz2.BZ2Decompressor()

        while True:
            chunk = compressed_file.read(CHUNK_SIZE)
            if not chunk:
                break

            decompressed = decompressor.decompress(chunk)
            decompressed_file.write(decompressed)
            pbar.update(len(chunk))


if __name__ == "__main__":
    URL = (
        "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    )

    BZ2_FILE = "/mnt/ai/data/en-wiki/enwiki-latest-pages-articles.xml.bz2"
    XML_FILE = "/mnt/ai/data/en-wiki/enwiki-latest-pages-articles.xml"

    # Step 1: Download
    if not os.path.exists(BZ2_FILE):
        download_with_progress(URL, BZ2_FILE)
    else:
        print(f"{BZ2_FILE} already exists, skipping download.")

    # Step 2: Decompress
    if not os.path.exists(XML_FILE):
        decompress_bz2_with_progress(BZ2_FILE, XML_FILE)
    else:
        print(f"{XML_FILE} already exists, skipping decompression.")
