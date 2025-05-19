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
import datetime
import json
import base64
import soundfile as sf
import uuid

# Import our project modules
from core.transformer_generator import TransformerSampleGenerator
from core.improved_generator import ImprovedSampleGenerator
from core.processing_pipeline import ProcessingPipeline
from core.chopping_engine import ChoppingEngine

# Import the new tab UI functions
from tabs.generate_audio_ui import render_generate_audio_tab
from tabs.process_samples_ui import render_process_samples_tab
from tabs.train_model_ui import render_train_model_tab
from tabs.scrapers_ui import render_scrapers_tab
from tabs.manual_chopper_ui import render_manual_chopper_tab
from tabs.auto_chopper_ui import render_auto_chopper_tab

st.set_page_config(page_title="Chopper Audio Generator", page_icon="🔊", layout="wide")

st.title("🔊 Chopper Audio Generator")
st.markdown("""
This web app allows you to generate audio samples with ML models and collaborate on training.
Upload samples for processing, train a model, or generate new audio.
""")

# Initialize project stats in session state if not present
if 'project_stats' not in st.session_state:
    st.session_state.project_stats = {
        "raw_samples": 0,
        "processed_segments": 0,
        "models": 0,
        "generated_samples": 0
    }

# Initialize additional session states for training queue
if 'training_queue' not in st.session_state:
    st.session_state.training_queue = []
if 'queue_processing' not in st.session_state:
    st.session_state.queue_processing = False
if 'current_training_job' not in st.session_state:
    st.session_state.current_training_job = None
if 'user_id' not in st.session_state:
    # Generate a simple user ID for the session - in production this would use real auth
    st.session_state.user_id = str(uuid.uuid4())[:8]

# Update project stats
def update_project_stats():
    if os.path.exists("data/raw"):
        st.session_state.project_stats["raw_samples"] = len([
            f for f in Path("data/raw").glob('*') 
            if f.is_file() and f.suffix.lower() in ('.wav', '.mp3', '.ogg', '.flac', '.aac') # Added more types
        ])
    if os.path.exists("data/processed/segments"): # Corrected path from original
        st.session_state.project_stats["processed_segments"] = len(list(Path("data/processed/segments").glob('*.wav')))
    if os.path.exists("models"):
        st.session_state.project_stats["models"] = len(list(Path("models").glob('*.pt'))) # Only .pt models
    if os.path.exists("output"):
        st.session_state.project_stats["generated_samples"] = len([
            f for f in Path("output").glob('**/*.wav') # Search recursively in output
            if f.is_file()
        ])
update_project_stats()

# Project status bar
st.markdown("---")
status_col1, status_col2, status_col3, status_col4 = st.columns(4)
with status_col1:
    st.info(f"Raw samples: {st.session_state.project_stats['raw_samples']}")
with status_col2:
    st.info(f"Processed segments: {st.session_state.project_stats['processed_segments']}")
with status_col3:
    st.info(f"Models: {st.session_state.project_stats['models']}")
with status_col4:
    st.info(f"Generated samples: {st.session_state.project_stats['generated_samples']}")
st.markdown("---")

# Session state for storing data across reruns
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'processed_samples' not in st.session_state:
    st.session_state.processed_samples = []

# Sidebar with model selection
st.sidebar.header("Model Settings")
model_type = st.sidebar.selectbox(
    "Select model type",
    ["Transformer GAN", "Improved Generator"],
    key="main_model_type_select"
)

# Check for saved models
models_dir = Path("models")
if not models_dir.exists():
    os.makedirs(models_dir, exist_ok=True)

model_files = [f.name for f in models_dir.glob("*.pt")]
default_model_name = "transformer_gan.pt" if "transformer_gan.pt" in model_files else None
default_model_idx = 0 if default_model_name is None else model_files.index(default_model_name) + 1

model_file = st.sidebar.selectbox(
    "Select a saved model (if available)",
    ["None"] + model_files,
    index=default_model_idx,
    key="main_model_file_select"
)

# Function to load the selected model
@st.cache_resource
def load_model(m_type, m_path=None):
    if m_path and m_path != "None":
        full_path = os.path.join("models", m_path)
        if not os.path.exists(full_path):
            st.error(f"Model file not found: {full_path}")
            return None
        try:
            if m_type == "Transformer GAN":
                return TransformerSampleGenerator(model_path=full_path)
            else:
                return ImprovedSampleGenerator(model_path=full_path)
        except Exception as e:
            st.error(f"Error loading model {m_path}: {e}")
            return None
    else:
        try:
            if m_type == "Transformer GAN":
                return TransformerSampleGenerator()
            else:
                return ImprovedSampleGenerator()
        except Exception as e:
            st.error(f"Error initializing untrained {m_type} model: {e}")
            return None

# Main tabs
tab_names = ["Generate Audio", "Process Samples", "Train Model", "Scrapers", "Manual Chopper", "Auto Chopper"]
tabs = st.tabs(tab_names)

tab_map = {
    tab_names[0]: tabs[0],
    tab_names[1]: tabs[1],
    tab_names[2]: tabs[2],
    tab_names[3]: tabs[3],
    tab_names[4]: tabs[4],
    tab_names[5]: tabs[5],
}

with tab_map["Generate Audio"]:
    render_generate_audio_tab(load_model, model_type, model_file)

with tab_map["Process Samples"]:
    render_process_samples_tab()

with tab_map["Train Model"]:
    render_train_model_tab(model_type)

with tab_map["Scrapers"]:
    render_scrapers_tab()

with tab_map["Manual Chopper"]:
    render_manual_chopper_tab()

with tab_map["Auto Chopper"]:
    render_auto_chopper_tab()

# Footer
st.markdown("---")
st.write("Made with ❤️ by the Chopper Audio Team")
st.write("Version 0.1.0")

if __name__ == "__main__":
    # This is used when running the script as a standalone app
    pass
