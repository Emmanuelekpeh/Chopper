from playwright.sync_api import sync_playwright
import os
from typing import List

class PlaywrightScraper:
    """
    A scraper using Playwright to navigate Looperman.com and download free loops.
    Requires playwright and its browsers installed (`pip install playwright` + `playwright install`).
    """

    BASE_URL = 'https://www.looperman.com/loops/search'

    def __init__(self, download_dir: str = 'downloads'):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def search(self, query: str, pages: int = 1) -> List[str]:
        """
        Search Looperman for loops matching the query.
        Returns a list of individual loop page URLs.
        """
        results = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            for page_num in range(1, pages + 1):
                url = f"{self.BASE_URL}?search={query}&page={page_num}"
                page.goto(url)
                page.wait_for_selector('div.loop-item')
                items = page.query_selector_all('div.loop-item a.loopName')
                for a in items:
                    href = a.get_attribute('href')
                    if href and href.startswith('/loops/'):
                        full = f"https://www.looperman.com{href}"
                        results.append(full)
            browser.close()
        return results

    def download_loop(self, loop_url: str) -> str:
        """
        Navigate to a loop page and download the loop file.
        Returns the local file path of the downloaded loop.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(loop_url)
            # Wait for download button and initiate download
            with page.expect_download() as download_info:
                page.click('a#downloadFile')
            download = download_info.value
            path = download.path()
            filename = os.path.basename(download.suggested_filename)
            dest = os.path.join(self.download_dir, filename)
            download.save_as(dest)
            browser.close()
        return dest

    def bulk_download(self, query: str, pages: int = 1) -> List[str]:
        """
        Search and download loops for multiple pages of results.
        Returns a list of downloaded file paths.
        """
        urls = self.search(query, pages)
        downloaded = []
        for url in urls:
            try:
                filepath = self.download_loop(url)
                downloaded.append(filepath)
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
        return downloaded
