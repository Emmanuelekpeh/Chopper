import os
import requests
from typing import List, Dict

class ArchiveScraper:
    """
    A scraper to search and download audio samples from Archive.org using its public API.
    """

    SEARCH_URL = 'https://archive.org/advancedsearch.php'
    METADATA_URL = 'https://archive.org/metadata'
    DOWNLOAD_BASE = 'https://archive.org/download'

    def search(self, query: str, rows: int = 10) -> List[Dict]:
        """
        Search for audio items matching the query on Archive.org.

        Args:
            query (str): Search keywords.
            rows (int): Number of results to return.

        Returns:
            List[Dict]: List of item metadata (identifier, title).
        """
        params = {
            'q': f'({query}) AND mediatype:audio',
            'fl[]': ['identifier', 'title'],
            'rows': rows,
            'output': 'json'
        }
        resp = requests.get(self.SEARCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get('response', {}).get('docs', [])

    def download(self, identifier: str, dest_dir: str) -> str:
        """
        Download the first available audio file from an Archive.org item.

        Args:
            identifier (str): Archive.org item identifier.
            dest_dir (str): Directory to save the downloaded file.

        Returns:
            str: Local file path of the downloaded audio.
        """
        # Fetch metadata to list available files
        meta_url = f"{self.METADATA_URL}/{identifier}/files"
        resp = requests.get(meta_url)
        resp.raise_for_status()
        files = resp.json().get('result', [])

        # Find a suitable audio file
        for file_info in files:
            name = file_info.get('name', '')
            if name.lower().endswith(('.mp3', '.wav', '.ogg')):
                url = f"{self.DOWNLOAD_BASE}/{identifier}/{name}"
                response = requests.get(url, stream=True)
                response.raise_for_status()
                os.makedirs(dest_dir, exist_ok=True)
                file_path = os.path.join(dest_dir, name)
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return file_path
        raise RuntimeError(f"No supported audio files found for item '{identifier}'")
