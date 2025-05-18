import os
import requests
from typing import List, Dict

class SampleScraper:
    """
    A scraper to search and download audio samples from Freesound.org using its API.
    Requires an environment variable FREESOUND_API_KEY with your API token.
    """

    BASE_URL = 'https://freesound.org/apiv2'

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('FREESOUND_API_KEY')
        if not self.api_key:
            raise ValueError('FREESOUND_API_KEY not provided or set in environment')

    def search(self, query: str, page_size: int = 10) -> List[Dict]:
        """
        Search for samples matching the query.
        Returns list of sample metadata dicts.
        """
        url = f"{self.BASE_URL}/search/text/"
        params = {
            'query': query,
            'page_size': page_size,
            'token': self.api_key
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get('results', [])

    def download_preview(self, sample: Dict, dest_dir: str) -> str:
        """
        Download the best available preview of a sample to dest_dir.
        Returns the local file path.
        """
        previews = sample.get('previews', {})
        # Choose highest quality preview available
        preview_url = previews.get('preview-hq-mp3') or previews.get('preview-lq-mp3')
        if not preview_url:
            raise RuntimeError(f"No preview URL for sample {sample.get('id')}")
        response = requests.get(preview_url, stream=True)
        response.raise_for_status()
        os.makedirs(dest_dir, exist_ok=True)
        filename = f"{sample.get('id')}.mp3"
        filepath = os.path.join(dest_dir, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filepath

    def bulk_download(self, query: str, dest_dir: str, count: int = 20) -> List[str]:
        """
        Search for and download a number of sample previews matching the query.
        Returns list of local file paths.
        """
        results = self.search(query, page_size=count)
        downloaded = []
        for sample in results:
            try:
                path = self.download_preview(sample, dest_dir)
                downloaded.append(path)
            except Exception as e:
                print(f"Failed to download sample {sample.get('id')}: {e}")
        return downloaded
