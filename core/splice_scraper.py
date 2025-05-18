from playwright.sync_api import sync_playwright
import os
import re
import time
from pathlib import Path
from typing import List, Dict

class SpliceLoopScraper:
    """
    A scraper to download high-quality, royalty-free loops from Splice.
    Uses Playwright for browser automation.
    """

    BASE_URL = 'https://splice.com/sounds/splice/samples'
    LOGIN_URL = 'https://splice.com/login'
    
    def __init__(self, email=None, password=None, download_dir='data/raw'):
        self.email = email or os.getenv('SPLICE_EMAIL')
        self.password = password or os.getenv('SPLICE_PASSWORD')
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        if not (self.email and self.password):
            raise ValueError("Splice credentials required. Set SPLICE_EMAIL and SPLICE_PASSWORD env vars")
    
    def login(self, page):
        """Log into Splice account"""
        page.goto(self.LOGIN_URL)
        page.wait_for_selector('input[type="email"]')
        page.fill('input[type="email"]', self.email)
        page.fill('input[type="password"]', self.password)
        page.click('button[type="submit"]')
        page.wait_for_selector('.header-user-dropdown')  # Wait for login completion
    
    def search(self, query: str, genres: List[str] = None, bpm_range: List[int] = None) -> List[Dict]:
        """
        Search Splice for samples matching criteria
        
        Args:
            query: Search term
            genres: List of genres to filter by
            bpm_range: [min_bpm, max_bpm] range to filter by
            
        Returns:
            List of sample metadata dictionaries
        """
        results = []
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)  # Set to True in production
            page = browser.new_page()
            
            # Login first
            self.login(page)
            
            # Construct search URL with query
            search_url = f"{self.BASE_URL}?q={query}"
            
            # Add genres if specified
            if genres:
                for genre in genres:
                    search_url += f"&genres={genre}"
            
            # Add BPM range if specified
            if bpm_range and len(bpm_range) == 2:
                search_url += f"&bpm_min={bpm_range[0]}&bpm_max={bpm_range[1]}"
                
            # Go to search results
            page.goto(search_url)
            page.wait_for_selector('.sample-item')
            
            # Extract sample metadata
            items = page.query_selector_all('.sample-item')
            for item in items:
                try:
                    name = item.query_selector('.sample-name').inner_text()
                    bpm = item.query_selector('.sample-bpm').inner_text().replace('BPM', '').strip()
                    key = item.query_selector('.sample-key').inner_text().strip()
                    download_id = item.get_attribute('data-sample-id')
                    
                    results.append({
                        'id': download_id,
                        'name': name,
                        'bpm': bpm,
                        'key': key,
                        'url': f"https://splice.com/sounds/samples/{download_id}"
                    })
                except Exception as e:
                    print(f"Error extracting sample metadata: {e}")
            
            browser.close()
        
        return results
    
    def download_sample(self, sample_id: str) -> str:
        """
        Download a specific sample by ID
        
        Returns:
            Path to downloaded file
        """
        sample_url = f"https://splice.com/sounds/samples/{sample_id}"
        output_path = None
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)  # Set to True in production
            page = browser.new_page()
            
            # Configure download behavior
            page.context.set_default_timeout(60000)  # Increase timeout for downloads
            download_path = str(self.download_dir)
            page.context.tracing.start(screenshots=True, snapshots=True)
            
            # Login and navigate to sample page
            self.login(page)
            page.goto(sample_url)
            page.wait_for_selector('.download-button')
            
            # Start download
            with page.expect_download() as download_info:
                page.click('.download-button')
            
            download = download_info.value
            output_path = os.path.join(download_path, f"{sample_id}_{download.suggested_filename}")
            download.save_as(output_path)
            
            browser.close()
            
        return output_path
    
    def bulk_download(self, query: str, count: int = 20, genres: List[str] = None, 
                     bpm_range: List[int] = None) -> List[str]:
        """
        Search and download multiple samples
        
        Returns:
            List of paths to downloaded files
        """
        results = self.search(query, genres, bpm_range)
        downloaded = []
        
        # Limit to requested count
        to_download = results[:min(count, len(results))]
        
        for item in to_download:
            try:
                filepath = self.download_sample(item['id'])
                item['local_path'] = filepath
                downloaded.append(item)
                # Be nice to the server
                time.sleep(1)
            except Exception as e:
                print(f"Failed to download {item['id']}: {e}")
        
        return downloaded
