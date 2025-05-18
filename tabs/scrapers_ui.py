import streamlit as st
import os
from pathlib import Path

from core.playwright_scraper import PlaywrightScraper
from core.splice_scraper import SpliceLoopScraper
from core.processing_pipeline import ProcessingPipeline

def render_scrapers_tab():
    st.header("Sample Scrapers")

    scraper_type = st.selectbox(
        "Select scraper source",
        ["Looperman", "Splice"],
        key="scraper_type_select"
    )

    search_query = st.text_input("Search query", "drum loop", key="scraper_query")

    if scraper_type == "Looperman":
        num_pages = st.slider("Number of pages to scrape", 1, 10, 2, key="scraper_looperman_pages")
        if st.button("Scrape Looperman", key="scraper_looperman_button"):
            with st.spinner("Scraping samples from Looperman..."):
                try:
                    scraper = PlaywrightScraper(download_dir="data/raw")
                    samples = scraper.bulk_download(search_query, pages=num_pages)
                    if samples:
                        st.success(f"Downloaded {len(samples)} samples from Looperman")
                        st.subheader("Downloaded Samples (first 5)")
                        for i, sample_path in enumerate(samples[:5]):
                            sample_name = os.path.basename(sample_path)
                            st.write(f"{i+1}. {sample_name}")
                            if os.path.exists(sample_path): st.audio(sample_path)
                    else:
                        st.warning("No samples found matching your query.")
                except Exception as e:
                    st.error(f"Error scraping Looperman: {str(e)}")

    elif scraper_type == "Splice":
        num_samples = st.slider("Number of samples", 5, 50, 20, key="scraper_splice_samples")
        col1, col2 = st.columns(2)
        with col1:
            splice_email = st.text_input("Splice Email", type="password", key="scraper_splice_email")
        with col2:
            splice_password = st.text_input("Splice Password", type="password", key="scraper_splice_password")

        if st.button("Scrape Splice", key="scraper_splice_button"):
            if not splice_email or not splice_password:
                st.error("Splice login credentials required")
            else:
                with st.spinner("Scraping samples from Splice..."):
                    try:
                        os.environ["SPLICE_EMAIL"] = splice_email
                        os.environ["SPLICE_PASSWORD"] = splice_password
                        scraper = SpliceLoopScraper(download_dir="data/raw")
                        samples_info = scraper.bulk_download(search_query, count=num_samples)
                        if samples_info:
                            st.success(f"Downloaded {len(samples_info)} samples from Splice")
                            st.subheader("Downloaded Samples (first 5)")
                            for i, sample_data in enumerate(samples_info[:5]):
                                st.write(f"{i+1}. {sample_data.get('name', 'Unknown')} (BPM: {sample_data.get('bpm', 'N/A')})")
                                local_path = sample_data.get('local_path')
                                if local_path and os.path.exists(local_path): st.audio(local_path)
                        else:
                            st.warning("No samples found matching your query.")
                    except Exception as e:
                        st.error(f"Error scraping Splice: {str(e)}")

    if st.button("Process Scraped Samples", key="scraper_process_button"):
        with st.spinner("Processing newly scraped samples..."):
            sample_dir = Path("data/raw")
            # Ensure to only process audio files
            audio_extensions = ('.wav', '.mp3', '.ogg', '.flac', '.aac') # Add more if needed
            sample_paths = [str(f) for f in sample_dir.glob('*') 
                            if f.is_file() and f.suffix.lower() in audio_extensions]
            
            if sample_paths:
                pipeline = ProcessingPipeline()
                results = pipeline.process_batch(sample_paths)
                metadata = pipeline.create_dataset_metadata(results)
                st.session_state.processed_samples = metadata.get("segment_paths", [])
                st.success(f"Processed {metadata.get('successful', 0)} files successfully!")
                st.info(f"Created {metadata.get('total_segments', 0)} segments.")
            else:
                st.warning("No audio files found in data/raw directory to process.") 