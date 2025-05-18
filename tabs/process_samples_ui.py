import streamlit as st
import os
import librosa
import matplotlib.pyplot as plt
from core.processing_pipeline import ProcessingPipeline # Ensure this import is correct

def render_process_samples_tab():
    st.header("Process Audio Samples")

    st.write("Upload audio samples for processing:")
    uploaded_files = st.file_uploader("Choose audio files", type=["wav", "mp3", "ogg"], accept_multiple_files=True, key="process_uploader")

    if uploaded_files:
        if st.button("Process Uploaded Samples"):
            os.makedirs("data/raw", exist_ok=True)
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data/raw", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)

            with st.spinner("Processing audio samples..."):
                pipeline = ProcessingPipeline() # This uses the imported class
                results = pipeline.process_batch(file_paths)
                metadata = pipeline.create_dataset_metadata(results)

                st.session_state.processed_samples = metadata.get("segment_paths", []) # Use .get for safety

                st.success(f"Processed {metadata.get('successful', 0)} files successfully!")
                st.info(f"Created {metadata.get('total_segments', 0)} segments.")

                if st.session_state.processed_samples:
                    st.write("Sample segments:")
                    for i, segment_path in enumerate(st.session_state.processed_samples[:3]):
                        try:
                            waveform, sr = librosa.load(segment_path, sr=None)
                            fig, ax = plt.subplots(figsize=(10, 2))
                            ax.plot(waveform)
                            ax.set_title(f"Segment {i+1}")
                            st.pyplot(fig)
                            st.audio(segment_path)
                        except Exception as e:
                            st.error(f"Error displaying segment {segment_path}: {e}") 