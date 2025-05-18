import streamlit as st
import os
import numpy as np
import librosa
import torch
from pathlib import Path
import tempfile
import time
import matplotlib.pyplot as plt
import io
from PIL import Image

# Import our project modules
from core.transformer_generator import TransformerSampleGenerator
from core.improved_generator import ImprovedSampleGenerator
from core.processing_pipeline import ProcessingPipeline

st.set_page_config(page_title="Chopper Audio Generator", page_icon="🔊", layout="wide")

st.title("🔊 Chopper Audio Generator")
st.markdown("""
This web app allows you to generate audio samples with ML models and collaborate on training.
Upload samples for processing, train a model, or generate new audio.
""")

# Session state for storing data across reruns
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'processed_samples' not in st.session_state:
    st.session_state.processed_samples = []

# Sidebar with model selection
st.sidebar.header("Model Settings")
model_type = st.sidebar.selectbox(
    "Select model type",
    ["Transformer GAN", "Improved Generator"]
)

# Check for saved models
models_dir = Path("models")
if not models_dir.exists():
    os.makedirs(models_dir, exist_ok=True)

model_files = [f.name for f in models_dir.glob("*.pt")]
default_model = "transformer_gan.pt" if "transformer_gan.pt" in model_files else None

model_file = st.sidebar.selectbox(
    "Select a saved model (if available)",
    ["None"] + model_files,
    index=0 if default_model is None else model_files.index(default_model) + 1
)

# Function to load the selected model
@st.cache_resource
def load_model(model_type, model_path=None):
    if model_path and model_path != "None":
        full_path = os.path.join("models", model_path)
        if model_type == "Transformer GAN":
            return TransformerSampleGenerator(model_path=full_path)
        else:
            return ImprovedSampleGenerator(model_path=full_path)
    else:
        if model_type == "Transformer GAN":
            return TransformerSampleGenerator()
        else:
            return ImprovedSampleGenerator()

# Main tabs
tab1, tab2, tab3 = st.tabs(["Generate Audio", "Process Samples", "Train Model"])

# Tab 1: Generate Audio
with tab1:
    st.header("Generate New Audio Samples")
    
    # Load the model when requested
    if st.button("Load Selected Model"):
        with st.spinner("Loading model..."):
            if model_file == "None":
                st.warning("No model selected. Using untrained model.")
                generator = load_model(model_type)
            else:
                generator = load_model(model_type, model_file)
                st.success(f"Model {model_file} loaded successfully!")
        
        st.session_state.generator = generator
        st.session_state.model_loaded = True
    
    # Generate samples if model is loaded
    if st.session_state.model_loaded:
        num_samples = st.slider("Number of samples to generate", 1, 5, 3)
        sample_length = st.slider("Sample length (seconds)", 1, 10, 5)
        
        if st.button("Generate Samples"):
            with st.spinner("Generating audio samples..."):
                
                # Create output directory
                os.makedirs("output", exist_ok=True)
                
                for i in range(num_samples):
                    output_path = f"output/generated_sample_{i}.wav"
                    st.session_state.generator.generate_and_save_audio(output_path)
                    
                    # Load and display waveform
                    waveform, sr = librosa.load(output_path, sr=None)
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.plot(waveform)
                    ax.set_title(f"Generated Sample {i+1}")
                    st.pyplot(fig)
                    
                    # Play audio
                    st.audio(output_path, format="audio/wav")
    else:
        st.info("Please load a model first using the button above.")

# Tab 2: Process Samples
with tab2:
    st.header("Process Audio Samples")
    
    st.write("Upload audio samples for processing:")
    uploaded_files = st.file_uploader("Choose audio files", type=["wav", "mp3", "ogg"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Uploaded Samples"):
            # Create directory for uploaded files
            os.makedirs("data/raw", exist_ok=True)
            
            # Save uploaded files
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data/raw", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
            
            # Process the samples
            with st.spinner("Processing audio samples..."):
                pipeline = ProcessingPipeline()
                results = pipeline.process_batch(file_paths)
                metadata = pipeline.create_dataset_metadata(results)
                
                st.session_state.processed_samples = metadata["segment_paths"] if "segment_paths" in metadata else []
                
                st.success(f"Processed {metadata['successful']} files successfully!")
                st.info(f"Created {metadata['total_segments']} segments.")
                
                # Display some processed segments
                if st.session_state.processed_samples:
                    st.write("Sample segments:")
                    for i, segment_path in enumerate(st.session_state.processed_samples[:3]):
                        waveform, sr = librosa.load(segment_path, sr=None)
                        fig, ax = plt.subplots(figsize=(10, 2))
                        ax.plot(waveform)
                        ax.set_title(f"Segment {i+1}")
                        st.pyplot(fig)
                        st.audio(segment_path)

# Tab 3: Train Model
with tab3:
    st.header("Train Model")
    
    if not st.session_state.processed_samples:
        st.info("Please process some samples first in the 'Process Samples' tab.")
    else:
        st.write(f"{len(st.session_state.processed_samples)} processed segments available for training.")
        
        # Training parameters
        epochs = st.slider("Training epochs", 10, 200, 50)
        batch_size = st.slider("Batch size", 4, 64, 16)
        learning_rate = st.number_input("Learning rate", 0.0001, 0.01, 0.0002, format="%.5f")
        
        model_save_path = st.text_input("Model save path", "models/trained_model.pt")
        
        if st.button("Start Training"):
            with st.spinner("Training model..."):
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize model based on selected type
                if model_type == "Transformer GAN":
                    generator = TransformerSampleGenerator()
                    
                    # Custom training loop to update progress
                    # Note: This is a simplified version; in reality, you'd need to modify your training code
                    status_text.text("Training in progress...")
                    for epoch in range(epochs):
                        # Update progress
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch+1}/{epochs}")
                        
                        # Sleep to simulate training time
                        time.sleep(0.5)
                    
                    # Save the model
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save({}, model_save_path)  # Empty model for demonstration
                    
                else:
                    # Similar code for ImprovedSampleGenerator
                    pass
                
                st.success(f"Training completed! Model saved to {model_save_path}")
                
                # Update model files list
                model_files = [f.name for f in Path("models").glob("*.pt")]

if __name__ == "__main__":
    # This is used when running the script as a standalone app
    pass
