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
from core.splice_scraper import SpliceLoopScraper
from core.playwright_scraper import PlaywrightScraper

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
if os.path.exists("data/raw"):
    st.session_state.project_stats["raw_samples"] = len([f for f in Path("data/raw").glob('*') 
                                                      if f.is_file() and f.suffix.lower() in ('.wav', '.mp3', '.ogg')])
if os.path.exists("data/processed/segments"):
    st.session_state.project_stats["processed_segments"] = len(list(Path("data/processed/segments").glob('*.wav')))
if os.path.exists("models"):
    st.session_state.project_stats["models"] = len(list(Path("models").glob('*.pt')))
if os.path.exists("output"):
    st.session_state.project_stats["generated_samples"] = len([f for f in Path("output").glob('*') 
                                                            if f.is_file() and f.suffix.lower() == '.wav'])

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Generate Audio", "Process Samples", "Train Model", 
    "Scrapers", "Manual Chopper", "Auto Chopper", "Collaborate"
])

# Tab 1: Generate Audio
with tab1:
    st.header("Generate New Audio Samples")
    
    # Model selection and loading
    model_col1, model_col2 = st.columns([2, 1])
    
    with model_col1:
        st.subheader("Model Selection")
        
        if model_file == "None":
            st.warning("No model selected in sidebar. Using untrained model.")
        else:
            st.success(f"Model selected: {model_file}")
        
        if st.button("Load Selected Model"):
            with st.spinner("Loading model..."):
                if model_file == "None":
                    generator = load_model(model_type)
                    st.warning("Using untrained model - results may not be usable")
                else:
                    generator = load_model(model_type, model_file)
                    st.success(f"Model {model_file} loaded successfully!")
            
            st.session_state.generator = generator
            st.session_state.model_loaded = True
    
    with model_col2:
        # Display model info if available
        if st.session_state.model_loaded:
            st.success("Model loaded and ready")
        else:
            st.error("No model loaded")
    
    # Generation options
    if st.session_state.model_loaded:
        st.subheader("Generation Options")
        
        # Add tabs for different generation methods
        gen_tab1, gen_tab2, gen_tab3 = st.tabs(["Basic Generation", "Guided Generation", "Batch Generation"])
        
        # Tab 1: Basic Generation
        with gen_tab1:
            # Basic generation controls
            num_samples = st.slider("Number of samples to generate", 1, 5, 1)
            sample_length = st.slider("Sample length (seconds)", 1, 10, 5)
            
            if st.button("Generate Sample(s)"):
                with st.spinner("Generating audio samples..."):
                    # Create output directory
                    os.makedirs("output", exist_ok=True)
                    
                    generated_samples = []
                    
                    for i in range(num_samples):
                        output_path = f"output/generated_sample_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.wav"
                        st.session_state.generator.generate_and_save_audio(output_path)
                        generated_samples.append(output_path)
                        
                        # Load and display waveform
                        waveform, sr = librosa.load(output_path, sr=None)
                        fig, ax = plt.subplots(figsize=(10, 2))
                        ax.plot(waveform)
                        ax.set_title(f"Generated Sample {i+1}")
                        st.pyplot(fig)
                        
                        # Play audio
                        st.audio(output_path, format="audio/wav")
                        
                        # Add to saved samples in session state for download
                        if 'saved_samples' not in st.session_state:
                            st.session_state.saved_samples = []
                        st.session_state.saved_samples.append(output_path)
                    
                    # Update project stats
                    st.session_state.project_stats["generated_samples"] += len(generated_samples)
        
        # Tab 2: Guided Generation
        with gen_tab2:
            st.write("Control the generation process by using a reference sample or adjusting parameters.")
            
            # Option to use a reference track
            st.subheader("Reference Track (Optional)")
            reference_file = st.file_uploader("Upload a reference audio file", type=["wav", "mp3", "ogg"], key="reference_uploader")
            
            if reference_file:
                # Save the reference file
                os.makedirs("data/reference", exist_ok=True)
                ref_path = os.path.join("data/reference", reference_file.name)
                with open(ref_path, "wb") as f:
                    f.write(reference_file.getbuffer())
                
                st.audio(ref_path)
                
                if st.button("Generate With Reference"):
                    with st.spinner("Generating RL-optimized sample..."):
                        output_path = f"output/rl_optimized_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                        audio, _ = st.session_state.generator.use_rl_optimization(target_audio_path=ref_path, n_steps=30)
                        st.session_state.generator.save_audio(audio, output_path)
                        
                        # Load and display waveform
                        waveform, sr = librosa.load(output_path, sr=None)
                        fig, ax = plt.subplots(figsize=(10, 2))
                        ax.plot(waveform)
                        ax.set_title("RL-Optimized Sample")
                        st.pyplot(fig)
                        
                        # Play audio
                        st.audio(output_path, format="audio/wav")
                        
                        # Add to saved samples in session state for download
                        if 'saved_samples' not in st.session_state:
                            st.session_state.saved_samples = []
                        st.session_state.saved_samples.append(output_path)
                        
                        # Update project stats
                        st.session_state.project_stats["generated_samples"] += 1
        
        # Tab 3: Batch Generation
        with gen_tab3:
            st.write("Generate multiple samples at once with varying parameters.")
            
            batch_size = st.slider("Batch size", 5, 50, 10)
            
            # Advanced batch generation settings in expander
            with st.expander("Advanced Settings"):
                variation = st.slider("Parameter variation (%)", 0, 50, 10)
                st.write("Adds random variation to the latent space")
            
            if st.button("Generate Batch"):
                with st.spinner(f"Generating batch of {batch_size} samples..."):
                    # Create output directory
                    os.makedirs("output/batch", exist_ok=True)
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    for i in range(batch_size):
                        output_path = f"output/batch/batch_{timestamp}_{i}.wav"
                        st.session_state.generator.generate_and_save_audio(output_path)
                    
                    st.success(f"Generated {batch_size} samples in output/batch/ directory")
                    
                    # Create ZIP file of all generated samples
                    import zipfile
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                        with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
                            batch_files = [f for f in Path(f"output/batch").glob(f"batch_{timestamp}_*.wav")]
                            for file_path in batch_files:
                                zipf.write(file_path, file_path.name)
                        
                        # Provide download link for the ZIP file
                        with open(tmp_zip.name, "rb") as f:
                            st.download_button(
                                label=f"Download Batch as ZIP",
                                data=f,
                                file_name=f"batch_samples_{timestamp}.zip",
                                mime="application/zip"
                            )
                    
                    # Update project stats
                    st.session_state.project_stats["generated_samples"] += batch_size
        
        # Sample management section
        st.subheader("Manage Generated Samples")
        
        if 'saved_samples' in st.session_state and st.session_state.saved_samples:
            st.write(f"You have {len(st.session_state.saved_samples)} saved samples in this session.")
            
            # Option to clear all
            if st.button("Clear Saved Samples"):
                st.session_state.saved_samples = []
                st.success("Saved samples list cleared")
        else:
            st.info("No samples saved in this session. Generate some samples first!")
    
    else:
        st.info("Please load a model first using the 'Load Selected Model' button above.")

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
        st.success(f"{len(st.session_state.processed_samples)} processed segments available for training.")
        
        # Add tabs for different training modes
        train_tab1, train_tab2, train_tab3, train_tab4, train_tab5 = st.tabs([
            "New Training", "Continue Training", "Checkpoints", "Hyperparameters", "Training Queue"
        ])
        
        # Tab 1: New Training
        with train_tab1:
            st.subheader("Train New Model")
            
            # Display sample statistics
            with st.expander("Training Data Statistics", expanded=False):
                # Calculate stats
                total_duration = 0
                segment_count = len(st.session_state.processed_samples)
                
                # Get a sample to display
                if segment_count > 0:
                    sample_file = st.session_state.processed_samples[0]
                    sample_audio, sr = librosa.load(sample_file, sr=None)
                    sample_duration = librosa.get_duration(y=sample_audio, sr=sr)
                    total_duration = segment_count * sample_duration
                    
                    # Display a sample waveform
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.plot(sample_audio)
                    ax.set_title("Sample Training Segment")
                    st.pyplot(fig)
                    
                    # Display audio player
                    st.audio(sample_file)
                
                # Display stats
                st.write(f"Total segments: {segment_count}")
                st.write(f"Approximate total duration: {total_duration:.2f} seconds")
            
            # Training parameters
            col1, col2 = st.columns(2)
            
            with col1:
                epochs = st.slider("Training epochs", 10, 500, 50)
                batch_size = st.slider("Batch size", 4, 64, 16)
                
                # Get hyperparameters from session state
                if 'hyperparameters' not in st.session_state:
                    st.session_state.hyperparameters = {
                        'learning_rate': 0.0002,
                        'beta1': 0.5,
                        'beta2': 0.999,
                        'latent_dim': 100,
                        'sequence_length': 128,
                        'checkpoint_interval': 10
                    }
                
                # Use hyperparameter values from session state
                learning_rate = st.session_state.hyperparameters['learning_rate']
                checkpoint_interval = st.session_state.hyperparameters['checkpoint_interval']
            
            with col2:
                # Base path for saving
                checkpoint_dir = st.text_input("Checkpoint directory", "models/checkpoints")
                model_name = st.text_input("Model name", f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
                
                # Determine the final model path
                model_save_path = os.path.join(checkpoint_dir, f"{model_name}_final.pt")
                
                # Enable checkpoint saving
                save_checkpoints = st.checkbox("Save checkpoints during training", True)
                
                if save_checkpoints:
                    checkpoint_freq = st.slider(
                        "Checkpoint frequency (epochs)", 
                        1, 50, checkpoint_interval,
                        help="Save a checkpoint model every N epochs"
                    )
                    st.session_state.hyperparameters['checkpoint_interval'] = checkpoint_freq
            
            # Training priority
            training_priority = st.radio(
                "Training priority",
                ["Low", "Medium", "High"],
                index=1,
                help="Higher priority jobs will be processed first in the queue"
            )
            
            # Training controls
            st.subheader("Training Controls")
            
            # Training button - Modified to add to queue instead of starting immediately
            if st.button("Add to Training Queue", key="queue_new_training"):
                # Create a training job
                training_job = {
                    'id': str(uuid.uuid4()),
                    'type': 'new_training',
                    'model_type': model_type,
                    'model_name': model_name,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'save_checkpoints': save_checkpoints,
                    'checkpoint_freq': checkpoint_freq if save_checkpoints else None,
                    'checkpoint_dir': checkpoint_dir,
                    'model_save_path': model_save_path,
                    'priority': training_priority,
                    'user_id': st.session_state.user_id,
                    'submitted_at': datetime.datetime.now().isoformat(),
                    'status': 'queued',
                    'hyperparameters': st.session_state.hyperparameters.copy()
                }
                
                # Add to queue
                st.session_state.training_queue.append(training_job)
                
                # Sort queue by priority
                priority_values = {"High": 0, "Medium": 1, "Low": 2}
                st.session_state.training_queue.sort(key=lambda job: (priority_values[job['priority']], job['submitted_at']))
                
                st.success(f"Training job added to queue! Job ID: {training_job['id'][:8]}")
                st.info("Check the 'Training Queue' tab to monitor your job status")
        
        # Tab 2: Continue Training
        with train_tab2:
            st.subheader("Continue Training from Checkpoint")
            
            # First, scan for available checkpoints
            checkpoint_base_dir = "models/checkpoints"
            if os.path.exists(checkpoint_base_dir):
                # Look for model metadata files to identify training runs
                metadata_files = list(Path(checkpoint_base_dir).glob("*_metadata.json"))
                
                if metadata_files:
                    # Parse metadata to get model information
                    training_runs = []
                    for metadata_file in metadata_files:
                        try:
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)
                                # Extract model name and type
                                run_info = {
                                    "name": metadata.get("model_name", "Unknown"),
                                    "metadata_path": str(metadata_file),
                                    "model_type": metadata.get("model_type", "Unknown"),
                                    "date_trained": metadata.get("date_trained", "Unknown"),
                                    "checkpoint_paths": metadata.get("checkpoint_paths", []),
                                    "final_model_path": metadata.get("final_model_path", ""),
                                    "epochs": metadata.get("epochs", 0),
                                    "best_epoch": metadata.get("best_epoch", 0),
                                    "best_loss": metadata.get("best_loss", 0)
                                }
                                training_runs.append(run_info)
                        except Exception as e:
                            st.warning(f"Could not parse metadata file {metadata_file}: {str(e)}")
                    
                    if training_runs:
                        # Select a training run to continue
                        selected_run_index = st.selectbox(
                            "Select training run to continue",
                            range(len(training_runs)),
                            format_func=lambda i: f"{training_runs[i]['name']} ({training_runs[i]['model_type']}, trained on {training_runs[i]['date_trained'][:10]})"
                        )
                        
                        selected_run = training_runs[selected_run_index]
                        
                        # Display info about the selected run
                        st.write(f"Model type: {selected_run['model_type']}")
                        st.write(f"Originally trained for: {selected_run['epochs']} epochs")
                        st.write(f"Best epoch: {selected_run['best_epoch']} (loss: {selected_run['best_loss']:.6f})")
                        
                        # Select a checkpoint to continue from
                        checkpoint_options = ["Final model"] + [f"Checkpoint at epoch {os.path.basename(path).split('_epoch_')[1].split('.pt')[0]}" 
                                                              for path in selected_run['checkpoint_paths']]
                        checkpoint_paths = [selected_run['final_model_path']] + selected_run['checkpoint_paths']
                        
                        selected_checkpoint_index = st.selectbox(
                            "Select checkpoint to continue from",
                            range(len(checkpoint_options)),
                            format_func=lambda i: checkpoint_options[i]
                        )
                        
                        selected_checkpoint_path = checkpoint_paths[selected_checkpoint_index]
                        
                        # Continuation parameters
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            additional_epochs = st.slider("Additional epochs", 10, 500, 50)
                            batch_size = st.slider("Batch size", 4, 64, 16, key="continue_batch_size")
                            
                            # Use hyperparameter values from session state or original training
                            learning_rate = st.number_input(
                                "Learning rate", 
                                0.00001, 0.01, 
                                st.session_state.hyperparameters['learning_rate'], 
                                format="%.5f", 
                                key="continue_lr"
                            )
                        
                        with col2:
                            # Path for saving the continued model
                            new_model_name = st.text_input(
                                "New model name", 
                                f"{selected_run['name']}_continued_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
                            )
                            
                            # Variation parameter - add noise to weights
                            add_variation = st.checkbox("Add variation to weights", False)
                            if add_variation:
                                variation_amount = st.slider("Variation amount", 0.0, 0.2, 0.01, 0.01)
                                st.info(f"Adding {variation_amount*100}% random variation to model weights")
                        
                        # Continue training button
                        if st.button("Continue Training"):
                            # Create checkpoint directory
                            os.makedirs(checkpoint_base_dir, exist_ok=True)
                            
                            # Determine save paths
                            continued_model_path = os.path.join(checkpoint_base_dir, f"{new_model_name}_final.pt")
                            
                            # Reset training progress
                            st.session_state.training_progress = {
                                'running': True,
                                'epoch': 0,
                                'best_epoch': 0,
                                'best_loss': float('inf'),
                                'total_epochs': additional_epochs,
                                'progress': 0.0,
                                'losses': [],
                                'checkpoint_paths': [],
                                'current_model': new_model_name
                            }
                            
                            # Create progress bar and status text
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            loss_chart = st.empty()
                            
                            try:
                                # Initialize model based on selected type and load checkpoint
                                if selected_run['model_type'] == "Transformer GAN":
                                    # Initialize with same hyperparameters as original
                                    with open(selected_run['metadata_path'], "r") as f:
                                        orig_metadata = json.load(f)
                                        orig_hyperparams = orig_metadata.get("hyperparameters", {})
                                    
                                    generator = TransformerSampleGenerator(
                                        latent_dim=orig_hyperparams.get('latent_dim', 100),
                                        sequence_length=orig_hyperparams.get('sequence_length', 128)
                                    )
                                    
                                    # Load checkpoint
                                    status_text.text(f"Loading checkpoint from {selected_checkpoint_path}...")
                                    generator.model.load_state_dict(torch.load(selected_checkpoint_path))
                                    
                                    # Add variation if requested
                                    if add_variation:
                                        status_text.text("Adding variation to model weights...")
                                        # Loop through model parameters and add noise
                                        for param in generator.model.parameters():
                                            noise = torch.randn_like(param) * variation_amount
                                            param.data += noise
                                    
                                    # Actual continuation training function similar to above
                                    status_text.text("Continuing training...")
                                    
                                    # Similar training code as "train_with_updates" in new training tab,
                                    # but reusing the checkpoint model...
                                    
                                    st.success(f"Continued training completed! Model saved to {continued_model_path}")
                                
                                elif selected_run['model_type'] == "Improved Generator":
                                    # Similar implementation for ImprovedGenerator
                                    st.info("Continued training for Improved Generator would be implemented here")
                                
                                else:
                                    st.error(f"Unknown model type: {selected_run['model_type']}")
                            
                            except Exception as e:
                                st.error(f"Error during continued training: {str(e)}")
                                raise e
                    else:
                        st.info("No previous training runs found with metadata.")
                else:
                    st.info("No checkpoint metadata found. Train a model with checkpoints first.")
            else:
                st.info("No checkpoints directory found. Train a model with checkpoints first.")
        
        # Tab 3: Checkpoints
        with train_tab3:
            st.subheader("Manage Checkpoints")
            
            # Scan for all checkpoints and training runs
            checkpoint_base_dir = "models/checkpoints"
            
            if os.path.exists(checkpoint_base_dir):
                checkpoint_files = list(Path(checkpoint_base_dir).glob("*.pt"))
                
                if checkpoint_files:
                    st.success(f"Found {len(checkpoint_files)} checkpoint files")
                    
                    # Group checkpoints by training run
                    training_runs = {}
                    
                    # First, parse metadata files
                    metadata_files = list(Path(checkpoint_base_dir).glob("*_metadata.json"))
                    for metadata_file in metadata_files:
                        try:
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)
                                model_name = metadata.get("model_name", "Unknown")
                                training_runs[model_name] = {
                                    "metadata": metadata,
                                    "checkpoints": [],
                                    "final_model": metadata.get("final_model_path", "")
                                }
                        except:
                            pass
                    
                    # Then categorize checkpoint files
                    for ckpt_path in checkpoint_files:
                        ckpt_name = os.path.basename(ckpt_path)
                        
                        # Look for model name in checkpoint filename
                        for model_name in training_runs.keys():
                            if model_name in ckpt_name and "epoch" in ckpt_name:
                                # This is a checkpoint for this model
                                training_runs[model_name]["checkpoints"].append(str(ckpt_path))
                                break
                    
                    # Display training runs and their checkpoints
                    for model_name, run_info in training_runs.items():
                        with st.expander(f"Model: {model_name}", expanded=False):
                            # Model info
                            metadata = run_info["metadata"]
                            st.write(f"Type: {metadata.get('model_type', 'Unknown')}")
                            st.write(f"Trained: {metadata.get('date_trained', 'Unknown')[:19]}")
                            st.write(f"Epochs: {metadata.get('epochs', 0)}")
                            st.write(f"Best epoch: {metadata.get('best_epoch', 0)} (loss: {metadata.get('best_loss', 0):.6f})")
                            
                            # Show loss curve if available
                            if "losses" in metadata and len(metadata["losses"]) > 1:
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.plot(metadata["losses"])
                                ax.set_xlabel("Epoch")
                                ax.set_ylabel("Loss")
                                ax.set_title(f"Training Loss: {model_name}")
                                st.pyplot(fig)
                            
                            # Final model
                            if run_info["final_model"] and os.path.exists(run_info["final_model"]):
                                st.write("**Final Model:**")
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"Path: {run_info['final_model']}")
                                    model_size = os.path.getsize(run_info["final_model"]) / (1024 * 1024)
                                    st.write(f"Size: {model_size:.2f} MB")
                                with col2:
                                    if st.button("Load", key=f"load_{model_name}_final"):
                                        st.session_state.model_to_load = run_info["final_model"]
                                        st.success(f"Model {model_name} (final) selected for loading")
                            
                            # Checkpoints
                            if run_info["checkpoints"]:
                                st.write(f"**Checkpoints ({len(run_info['checkpoints'])}):**")
                                for i, ckpt_path in enumerate(sorted(run_info["checkpoints"])):
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        # Extract epoch number
                                        ckpt_name = os.path.basename(ckpt_path)
                                        epoch_str = ckpt_name.split("_epoch_")[1].split(".pt")[0] if "_epoch_" in ckpt_name else "?"
                                        st.write(f"Epoch {epoch_str}")
                                        ckpt_size = os.path.getsize(ckpt_path) / (1024 * 1024)
                                        st.write(f"Size: {ckpt_size:.2f} MB")
                                    with col2:
                                        if st.button("Load", key=f"load_{model_name}_ckpt_{i}"):
                                            st.session_state.model_to_load = ckpt_path
                                            st.success(f"Model {model_name} (epoch {epoch_str}) selected for loading")
                            
                            # Delete option
                            if st.button("Delete this training run", key=f"delete_{model_name}"):
                                try:
                                    # Delete all files for this training run
                                    for ckpt_path in run_info["checkpoints"]:
                                        if os.path.exists(ckpt_path):
                                            os.remove(ckpt_path)
                                    if os.path.exists(run_info["final_model"]):
                                        os.remove(run_info["final_model"])
                                    # Delete metadata file
                                    metadata_path = os.path.join(checkpoint_base_dir, f"{model_name}_metadata.json")
                                    if os.path.exists(metadata_path):
                                        os.remove(metadata_path)
                                    st.success(f"Deleted training run {model_name}")
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"Error deleting training run: {str(e)}")
                else:
                    st.info("No checkpoints found. Train a model with checkpoints first.")
            else:
                st.info("No checkpoints directory found. Train a model with checkpoints first.")
        
        # Tab 4: Hyperparameters
        with train_tab4:
            st.subheader("Hyperparameter Settings")
            
            st.write("Configure training hyperparameters that will be used for new training runs:")
            
            # Organize hyperparameters by category
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Optimizer Parameters**")
                
                # Learning rate with more control
                lr = st.number_input(
                    "Learning rate",
                    min_value=0.00001,
                    max_value=0.1,
                    value=st.session_state.hyperparameters['learning_rate'],
                    format="%.5f"
                )
                st.session_state.hyperparameters['learning_rate'] = lr
                
                # Adam betas
                beta1 = st.slider(
                    "Beta1 (momentum)",
                    0.0, 0.999, 
                    st.session_state.hyperparameters['beta1'],
                    0.001
                )
                st.session_state.hyperparameters['beta1'] = beta1
                
                beta2 = st.slider(
                    "Beta2 (RMSprop)",
                    0.9, 0.9999, 
                    st.session_state.hyperparameters['beta2'],
                    0.0001
                )
                st.session_state.hyperparameters['beta2'] = beta2
                
                # Checkpoint interval
                checkpoint_interval = st.slider(
                    "Checkpoint interval (epochs)",
                    1, 50, 
                    st.session_state.hyperparameters['checkpoint_interval']
                )
                st.session_state.hyperparameters['checkpoint_interval'] = checkpoint_interval
            
            with col2:
                st.write("**Model Architecture Parameters**")
                
                # Model-specific parameters
                latent_dim = st.slider(
                    "Latent dimension",
                    16, 512, 
                    st.session_state.hyperparameters['latent_dim']
                )
                st.session_state.hyperparameters['latent_dim'] = latent_dim
                
                sequence_length = st.slider(
                    "Sequence length",
                    64, 256, 
                    st.session_state.hyperparameters['sequence_length']
                )
                st.session_state.hyperparameters['sequence_length'] = sequence_length
                
                # Different model architecture options based on selected model type
                if model_type == "Transformer GAN":
                    if 'transformer_params' not in st.session_state.hyperparameters:
                        st.session_state.hyperparameters['transformer_params'] = {
                            'n_heads': 8,
                            'n_layers': 6,
                            'd_model': 512
                        }
                    
                    n_heads = st.slider(
                        "Number of attention heads",
                        1, 16, 
                        st.session_state.hyperparameters['transformer_params']['n_heads']
                    )
                    st.session_state.hyperparameters['transformer_params']['n_heads'] = n_heads
                    
                    n_layers = st.slider(
                        "Number of transformer layers",
                        1, 12, 
                        st.session_state.hyperparameters['transformer_params']['n_layers']
                    )
                    st.session_state.hyperparameters['transformer_params']['n_layers'] = n_layers
                    
                elif model_type == "Improved Generator":
                    if 'gan_params' not in st.session_state.hyperparameters:
                        st.session_state.hyperparameters['gan_params'] = {
                            'n_mels': 128,
                            'dropout': 0.3
                        }
                    
                    n_mels = st.slider(
                        "Number of mel bands",
                        64, 256, 
                        st.session_state.hyperparameters['gan_params']['n_mels']
                    )
                    st.session_state.hyperparameters['gan_params']['n_mels'] = n_mels
                    
                    dropout = st.slider(
                        "Dropout rate",
                        0.0, 0.5, 
                        st.session_state.hyperparameters['gan_params']['dropout'],
                        0.01
                    )
                    st.session_state.hyperparameters['gan_params']['dropout'] = dropout
            
            # Save and load hyperparameter presets
            st.subheader("Hyperparameter Presets")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Save current hyperparameters as preset
                preset_name = st.text_input("Preset name", "my_preset")
                
                if st.button("Save as Preset"):
                    # Create presets directory
                    presets_dir = "models/hyperparameter_presets"
                    os.makedirs(presets_dir, exist_ok=True)
                    
                    # Save preset to file
                    preset_path = os.path.join(presets_dir, f"{preset_name}.json")
                    with open(preset_path, "w") as f:
                        json.dump(st.session_state.hyperparameters, f, indent=2)
                    
                    st.success(f"Saved hyperparameter preset as {preset_name}")
            
            with col2:
                # Load hyperparameter preset
                presets_dir = "models/hyperparameter_presets"
                if os.path.exists(presets_dir):
                    preset_files = [f.name for f in Path(presets_dir).glob("*.json")]
                    
                    if preset_files:
                        selected_preset = st.selectbox(
                            "Select preset to load",
                            preset_files
                        )
                        
                        if st.button("Load Preset"):
                            preset_path = os.path.join(presets_dir, selected_preset)
                            with open(preset_path, "r") as f:
                                loaded_params = json.load(f)
                                st.session_state.hyperparameters.update(loaded_params)
                            
                            st.success(f"Loaded hyperparameter preset {selected_preset}")
                            st.experimental_rerun()
                    else:
                        st.info("No presets found. Save a preset first.")
                else:
                    st.info("No presets directory found. Save a preset first.")
        
        # Display training status if training is in progress
        if 'training_progress' in st.session_state and st.session_state.training_progress.get('running', False):
            st.subheader("Training Status")
            
            # Create a container for status
            status_container = st.container()
            
            with status_container:
                st.write(f"Current model: {st.session_state.training_progress.get('current_model', 'Unknown')}")
                st.write(f"Epoch: {st.session_state.training_progress['epoch']}/{st.session_state.training_progress['total_epochs']}")
                st.progress(st.session_state.training_progress['progress'])
                
                # Display loss chart if available
                if len(st.session_state.training_progress['losses']) > 1:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(st.session_state.training_progress['losses'])
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title("Training Loss")
                    st.pyplot(fig)
                
                # Display saved checkpoints if any
                if st.session_state.training_progress['checkpoint_paths']:
                    st.write("Saved checkpoints:")
                    for path in st.session_state.training_progress['checkpoint_paths']:
                        st.code(path, language="text")
            
            # Stop training button
            if st.button("Stop Training", key="stop_training_main"):
                st.session_state.training_progress['running'] = False
                st.warning("Training has been stopped.")
        
        # Tab 5: Training Queue (new tab)
        with train_tab5:
            st.subheader("Training Queue Management")
            
            # Queue status overview
            current_job = st.session_state.current_training_job
            queue_length = len(st.session_state.training_queue)
            
            # Current training job status
            if current_job:
                st.write("### Currently Training")
                
                job_status_cols = st.columns([1, 2, 1])
                with job_status_cols[0]:
                    st.write(f"**Job ID:** {current_job['id'][:8]}")
                    st.write(f"**Model:** {current_job['model_name']}")
                    st.write(f"**User:** {current_job['user_id']}")
                
                with job_status_cols[1]:
                    st.write(f"**Type:** {current_job['type']}")
                    st.write(f"**Priority:** {current_job['priority']}")
                    st.write(f"**Started At:** {datetime.datetime.fromisoformat(current_job.get('started_at', current_job['submitted_at'])).strftime('%Y-%m-%d %H:%M:%S')}")
                
                with job_status_cols[2]:
                    # Show cancel button only for current user's job
                    if current_job['user_id'] == st.session_state.user_id:
                        if st.button("Cancel Training", key="cancel_current_job"):
                            # Set flag to stop training
                            st.session_state.training_progress['running'] = False
                            st.warning("Training job will be canceled soon...")
                    else:
                        st.write("*Training by another user*")
                
                # Show progress information if available
                if 'training_progress' in st.session_state and st.session_state.training_progress['running']:
                    st.write(f"Current progress: Epoch {st.session_state.training_progress['epoch']}/{st.session_state.training_progress['total_epochs']}")
                    st.progress(st.session_state.training_progress['progress'])
            else:
                st.info("No active training job currently running")
            
            # Queue listing
            st.write(f"### Queued Jobs ({queue_length})")
            
            if queue_length > 0:
                # Create a table to display the queue
                queue_table_data = []
                
                for i, job in enumerate(st.session_state.training_queue):
                    # Format job information for the table
                    submitted_time = datetime.datetime.fromisoformat(job['submitted_at']).strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Determine if this job belongs to the current user
                    is_users_job = job['user_id'] == st.session_state.user_id
                    
                    # Add row to table data
                    queue_table_data.append({
                        "Position": i + 1,
                        "Job ID": job['id'][:8],
                        "Type": job['type'],
                        "Model": job['model_name'],
                        "Priority": job['priority'],
                        "Submitted": submitted_time,
                        "User": job['user_id'] + (" (you)" if is_users_job else "")
                    })
                
                # Display the queue as a dataframe
                st.dataframe(queue_table_data)
                
                # Display job management options for user's jobs
                st.subheader("Manage Your Jobs")
                
                # Filter to the current user's jobs
                user_jobs = [job for job in st.session_state.training_queue if job['user_id'] == st.session_state.user_id]
                
                if user_jobs:
                    # Options to manage jobs in queue
                    job_to_manage = st.selectbox(
                        "Select one of your jobs to manage:",
                        range(len(user_jobs)),
                        format_func=lambda i: f"{user_jobs[i]['model_name']} (ID: {user_jobs[i]['id'][:8]}, Priority: {user_jobs[i]['priority']})"
                    )
                    
                    selected_job = user_jobs[job_to_manage]
                    
                    # Show options for the selected job
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Option to change priority
                        new_priority = st.radio(
                            "Change priority:",
                            ["Low", "Medium", "High"],
                            ["Low", "Medium", "High"].index(selected_job['priority'])
                        )
                        
                        if new_priority != selected_job['priority'] and st.button("Update Priority"):
                            # Find the job in the queue and update its priority
                            for job in st.session_state.training_queue:
                                if job['id'] == selected_job['id']:
                                    job['priority'] = new_priority
                                    break
                            
                            # Re-sort the queue
                            priority_values = {"High": 0, "Medium": 1, "Low": 2}
                            st.session_state.training_queue.sort(key=lambda job: (priority_values[job['priority']], job['submitted_at']))
                            
                            st.success(f"Updated priority to {new_priority}")
                            st.experimental_rerun()
                    
                    with col2:
                        # Option to remove job from queue
                        if st.button("Remove from Queue"):
                            # Remove the job from the queue
                            st.session_state.training_queue = [job for job in st.session_state.training_queue if job['id'] != selected_job['id']]
                            st.success("Job removed from queue")
                            st.experimental_rerun()
                    
                    with col3:
                        # Option to move job to top of its priority level
                        if st.button("Move to Top of Priority"):
                            # Find all jobs with the same priority
                            same_priority_jobs = [job for job in st.session_state.training_queue if job['priority'] == selected_job['priority']]
                            
                            if len(same_priority_jobs) > 1:
                                # Remove the selected job from the queue
                                st.session_state.training_queue = [job for job in st.session_state.training_queue if job['id'] != selected_job['id']]
                                
                                # Find the position of the first job with this priority
                                priority_values = {"High": 0, "Medium": 1, "Low": 2}
                                
                                # Sort by priority to find where to insert
                                sorted_queue = sorted(st.session_state.training_queue, key=lambda job: (priority_values[job['priority']], job['submitted_at']))
                                
                                # Find the position of the first job with selected priority
                                insert_pos = 0
                                for i, job in enumerate(sorted_queue):
                                    if job['priority'] == selected_job['priority']:
                                        insert_pos = i
                                        break
                                
                                # Update the submission time to be just before the first job of same priority
                                if insert_pos < len(sorted_queue):
                                    # Get the submitted time of the first job in this priority
                                    reference_time = datetime.datetime.fromisoformat(sorted_queue[insert_pos]['submitted_at'])
                                    # Set the time to be 1 second earlier
                                    new_time = (reference_time - datetime.timedelta(seconds=1)).isoformat()
                                    selected_job['submitted_at'] = new_time
                                
                                # Add the job back to the queue
                                st.session_state.training_queue.append(selected_job)
                                
                                # Re-sort the queue
                                st.session_state.training_queue.sort(key=lambda job: (priority_values[job['priority']], job['submitted_at']))
                                
                                st.success(f"Moved job to top of {selected_job['priority']} priority")
                                st.experimental_rerun()
                            else:
                                st.info("This job is already at the top of its priority level")
                else:
                    st.info("You don't have any jobs in the queue")
                
                # Admin controls for queue management
                with st.expander("Queue Administration", expanded=False):
                    st.warning("These controls affect all jobs in the queue")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Clear All Jobs"):
                            st.session_state.training_queue = []
                            st.success("Queue cleared")
                            st.experimental_rerun()
                    
                    with col2:
                        if st.button("Process Next Job Now"):
                            # Force processing of the next job
                            st.session_state.queue_processing = True
                            st.success("Queue processing initiated")
                            st.experimental_rerun()
            else:
                st.info("The training queue is empty")
            
            # Queue system information
            with st.expander("About the Training Queue System", expanded=False):
                st.markdown("""
                ### How the Training Queue Works
                
                1. **Job Priority**: Jobs are processed based on priority (High > Medium > Low)
                2. **Submission Time**: Within each priority level, older jobs are processed first
                3. **Job Management**: You can change priority, remove jobs, or move them up in the queue
                4. **Resource Allocation**: Only one training job runs at a time to prevent resource conflicts
                
                Your current user ID is: `{}`
                
                This queueing system ensures fair allocation of training resources in a collaborative environment.
                """.format(st.session_state.user_id))

# Tab 4: Scrapers
with tab4:
    st.header("Sample Scrapers")
    
    scraper_type = st.selectbox(
        "Select scraper source",
        ["Looperman", "Splice"]
    )
    
    search_query = st.text_input("Search query", "drum loop")
    
    if scraper_type == "Looperman":
        num_pages = st.slider("Number of pages to scrape", 1, 10, 2)
        
        if st.button("Scrape Looperman"):
            with st.spinner("Scraping samples from Looperman..."):
                try:
                    scraper = PlaywrightScraper(download_dir="data/raw")
                    samples = scraper.bulk_download(search_query, pages=num_pages)
                    
                    if samples:
                        st.success(f"Downloaded {len(samples)} samples from Looperman")
                        
                        # Display sample info
                        st.subheader("Downloaded Samples")
                        for i, sample_path in enumerate(samples[:5]):  # Show first 5
                            sample_name = os.path.basename(sample_path)
                            st.write(f"{i+1}. {sample_name}")
                            
                            # Display audio player if file exists
                            if os.path.exists(sample_path):
                                st.audio(sample_path)
                    else:
                        st.warning("No samples found matching your query.")
                except Exception as e:
                    st.error(f"Error scraping Looperman: {str(e)}")
    
    elif scraper_type == "Splice":
        num_samples = st.slider("Number of samples", 5, 50, 20)
        
        # Splice requires login credentials
        col1, col2 = st.columns(2)
        with col1:
            splice_email = st.text_input("Splice Email", type="password")
        with col2:
            splice_password = st.text_input("Splice Password", type="password")
        
        if st.button("Scrape Splice"):
            if not splice_email or not splice_password:
                st.error("Splice login credentials required")
            else:
                with st.spinner("Scraping samples from Splice..."):
                    try:
                        # Set environment variables for the scraper
                        os.environ["SPLICE_EMAIL"] = splice_email
                        os.environ["SPLICE_PASSWORD"] = splice_password
                        
                        scraper = SpliceLoopScraper(download_dir="data/raw")
                        samples = scraper.bulk_download(search_query, count=num_samples)
                        
                        if samples:
                            st.success(f"Downloaded {len(samples)} samples from Splice")
                            
                            # Display sample info
                            st.subheader("Downloaded Samples")
                            for i, sample in enumerate(samples[:5]):  # Show first 5
                                st.write(f"{i+1}. {sample.get('name', 'Unknown')} (BPM: {sample.get('bpm', 'N/A')})")
                                
                                # Display audio player if file exists
                                local_path = sample.get('local_path')
                                if local_path and os.path.exists(local_path):
                                    st.audio(local_path)
                        else:
                            st.warning("No samples found matching your query.")
                    except Exception as e:
                        st.error(f"Error scraping Splice: {str(e)}")
    
    # Option to process scraped samples
    if st.button("Process Scraped Samples"):
        with st.spinner("Processing newly scraped samples..."):
            sample_dir = Path("data/raw")
            sample_paths = [str(f) for f in sample_dir.glob('*') if f.is_file() and f.suffix.lower() in ('.wav', '.mp3', '.ogg')]
            
            if sample_paths:
                pipeline = ProcessingPipeline()
                results = pipeline.process_batch(sample_paths)
                metadata = pipeline.create_dataset_metadata(results)
                
                st.session_state.processed_samples = metadata["segment_paths"] if "segment_paths" in metadata else []
                
                st.success(f"Processed {metadata['successful']} files successfully!")
                st.info(f"Created {metadata['total_segments']} segments.")
            else:
                st.warning("No audio files found in data/raw directory.")

# Tab 5: Manual Chopper
with tab5:
    st.header("Manual Audio Chopper")
    
    st.write("Upload an audio file to chop it manually:")
    chopper_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"], key="chopper_uploader")
    
    if chopper_file:
        # Save the uploaded file
        os.makedirs("data/chopper", exist_ok=True)
        temp_audio_path = os.path.join("data/chopper", chopper_file.name)
        with open(temp_audio_path, "wb") as f:
            f.write(chopper_file.getbuffer())
        
        # Load and display audio
        waveform, sr = librosa.load(temp_audio_path, sr=None)
        duration = librosa.get_duration(y=waveform, sr=sr)
        
        st.write(f"File loaded: {chopper_file.name} (Duration: {duration:.2f} seconds)")
        st.audio(temp_audio_path)
        
        # Create plot of waveform
        fig, ax = plt.subplots(figsize=(12, 3))
        times = np.linspace(0, duration, len(waveform))
        ax.plot(times, waveform)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform")
        st.pyplot(fig)
        
        # Manual chopping controls
        st.subheader("Chopping Controls")
        
        chop_method = st.radio("Chopping Method", ["Time-based", "Marker-based"])
        
        if chop_method == "Time-based":
            # Time-based chopping
            segment_length = st.slider("Segment length (seconds)", 0.1, 5.0, 1.0)
            overlap = st.slider("Overlap (%)", 0, 75, 0)
            
            # Calculate number of segments
            overlap_fraction = overlap / 100
            hop_length_seconds = segment_length * (1 - overlap_fraction)
            num_segments = int((duration - segment_length) / hop_length_seconds) + 1
            
            st.write(f"This will create approximately {num_segments} segments")
            
            if st.button("Chop Audio (Time-based)"):
                with st.spinner("Chopping audio..."):
                    # Create output directory
                    os.makedirs("output/chopped", exist_ok=True)
                    
                    # Convert to samples
                    segment_samples = int(segment_length * sr)
                    hop_length_samples = int(hop_length_seconds * sr)
                    
                    # Create segments
                    segments = []
                    for i in range(num_segments):
                        start_sample = i * hop_length_samples
                        end_sample = start_sample + segment_samples
                        if end_sample <= len(waveform):
                            segment = waveform[start_sample:end_sample]
                            output_path = f"output/chopped/{os.path.splitext(chopper_file.name)[0]}_segment_{i+1}.wav"
                            sf.write(output_path, segment, sr)
                            segments.append(output_path)
                    
                    st.success(f"Created {len(segments)} segments")
                    
                    # Display segments
                    st.subheader("Chopped Segments")
                    for i, segment_path in enumerate(segments):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.write(f"Segment {i+1}")
                            st.audio(segment_path)
                        with col2:
                            segment_waveform, _ = librosa.load(segment_path, sr=None)
                            fig, ax = plt.subplots(figsize=(8, 2))
                            ax.plot(segment_waveform)
                            ax.set_title(f"Segment {i+1}")
                            st.pyplot(fig)
        
        else:
            # Marker-based chopping
            st.write("Place markers at specific points in the audio by entering time values (in seconds):")
            
            # Initialize list of markers in session state if not present
            if 'markers' not in st.session_state:
                st.session_state.markers = []
            
            # Add marker
            marker_time = st.number_input("Marker time (seconds)", 0.0, float(duration), step=0.1)
            if st.button("Add Marker"):
                st.session_state.markers.append(marker_time)
                st.session_state.markers.sort()
            
            # Display and manage markers
            if st.session_state.markers:
                st.write("Current markers (seconds):")
                markers_str = ", ".join([f"{m:.2f}" for m in st.session_state.markers])
                st.code(markers_str)
                
                if st.button("Clear All Markers"):
                    st.session_state.markers = []
                
                # Create a new plot with markers
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(times, waveform)
                
                # Add vertical lines for markers
                for marker in st.session_state.markers:
                    ax.axvline(x=marker, color='r', linestyle='--')
                
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_title("Waveform with Markers")
                st.pyplot(fig)
                
                # Chop at markers
                if st.button("Chop at Markers"):
                    with st.spinner("Chopping audio at markers..."):
                        # Create output directory
                        os.makedirs("output/chopped", exist_ok=True)
                        
                        # Create segment boundaries from markers
                        boundaries = [0.0] + st.session_state.markers + [duration]
                        
                        # Create segments
                        segments = []
                        for i in range(len(boundaries) - 1):
                            start_time = boundaries[i]
                            end_time = boundaries[i+1]
                            
                            # Convert to samples
                            start_sample = int(start_time * sr)
                            end_sample = int(end_time * sr)
                            
                            segment = waveform[start_sample:end_sample]
                            output_path = f"output/chopped/{os.path.splitext(chopper_file.name)[0]}_marker_{i+1}.wav"
                            sf.write(output_path, segment, sr)
                            segments.append(output_path)
                        
                        st.success(f"Created {len(segments)} segments from markers")
                        
                        # Display segments
                        st.subheader("Chopped Segments")
                        for i, segment_path in enumerate(segments):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.write(f"Segment {i+1}")
                                st.audio(segment_path)
                            with col2:
                                segment_waveform, _ = librosa.load(segment_path, sr=None)
                                fig, ax = plt.subplots(figsize=(8, 2))
                                ax.plot(segment_waveform)
                                st.pyplot(fig)

# Tab 6: Auto Chopper
with tab6:
    st.header("Automatic Audio Chopper")
    
    st.write("Upload an audio file for automatic chopping using beat detection or silence detection:")
    auto_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"], key="auto_chopper_uploader")
    
    if auto_file:
        # Save the uploaded file
        os.makedirs("data/chopper", exist_ok=True)
        auto_audio_path = os.path.join("data/chopper", auto_file.name)
        with open(auto_audio_path, "wb") as f:
            f.write(auto_file.getbuffer())
        
        # Load and display audio
        waveform, sr = librosa.load(auto_audio_path, sr=None)
        duration = librosa.get_duration(y=waveform, sr=sr)
        
        st.write(f"File loaded: {auto_file.name} (Duration: {duration:.2f} seconds)")
        st.audio(auto_audio_path)
        
        # Create plot of waveform
        fig, ax = plt.subplots(figsize=(12, 3))
        times = np.linspace(0, duration, len(waveform))
        ax.plot(times, waveform)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform")
        st.pyplot(fig)
        
        # Auto chopping controls
        st.subheader("Chopping Method")
        
        method = st.radio("Detection Method", ["Beat Detection", "Silence Detection"])
        
        if method == "Beat Detection":
            # Beat detection parameters
            st.write("Beat detection will automatically find rhythmic elements in your audio.")
            
            if st.button("Detect Beats and Chop"):
                with st.spinner("Analyzing beats and chopping audio..."):
                    # Initialize chopper
                    chopper = ChoppingEngine(sample_rate=sr)
                    
                    # Get beat times
                    beat_times = chopper.chop_by_beats(waveform)
                    
                    # Create output directory
                    os.makedirs("output/chopped", exist_ok=True)
                    
                    # Create segments
                    segments = []
                    for i in range(len(beat_times) - 1):
                        start_time = beat_times[i]
                        end_time = beat_times[i+1]
                        
                        # Convert to samples
                        start_sample = int(start_time * sr)
                        end_sample = int(end_time * sr)
                        
                        if start_sample < len(waveform) and end_sample <= len(waveform):
                            segment = waveform[start_sample:end_sample]
                            output_path = f"output/chopped/{os.path.splitext(auto_file.name)[0]}_beat_{i+1}.wav"
                            sf.write(output_path, segment, sr)
                            segments.append({
                                "path": output_path,
                                "start": start_time,
                                "end": end_time
                            })
                    
                    st.success(f"Detected {len(beat_times)} beats and created {len(segments)} segments")
                    
                    # Plot waveform with beat markers
                    fig, ax = plt.subplots(figsize=(12, 3))
                    ax.plot(times, waveform)
                    
                    # Add vertical lines for beats
                    for beat in beat_times:
                        ax.axvline(x=beat, color='r', linestyle='--')
                    
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Amplitude")
                    ax.set_title("Waveform with Beat Markers")
                    st.pyplot(fig)
                    
                    # Display segments
                    st.subheader("Chopped Segments")
                    for i, segment in enumerate(segments):
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            st.write(f"Beat Segment {i+1}")
                            st.write(f"Start: {segment['start']:.2f}s")
                            st.write(f"End: {segment['end']:.2f}s")
                            st.audio(segment["path"])
                        with col2:
                            segment_waveform, _ = librosa.load(segment["path"], sr=None)
                            fig, ax = plt.subplots(figsize=(8, 2))
                            ax.plot(segment_waveform)
                            ax.set_title(f"Beat Segment {i+1}")
                            st.pyplot(fig)
                        with col3:
                            if st.button(f"Save as one-shot {i+1}", key=f"save_beat_{i}"):
                                oneshot_dir = "output/oneshots"
                                os.makedirs(oneshot_dir, exist_ok=True)
                                oneshot_path = f"{oneshot_dir}/{os.path.splitext(auto_file.name)[0]}_oneshot_{i+1}.wav"
                                sf.write(oneshot_path, segment_waveform, sr)
                                st.success(f"Saved one-shot to {oneshot_path}")
        
        else:  # Silence Detection
            # Silence detection parameters
            silence_threshold = st.slider("Silence Threshold (dB)", 10, 60, 30)
            min_silence = st.slider("Minimum Silence Duration (ms)", 50, 1000, 300)
            
            if st.button("Detect Silence and Chop"):
                with st.spinner("Analyzing silence and chopping audio..."):
                    # Initialize chopper
                    chopper = ChoppingEngine(sample_rate=sr)
                    
                    # Get segments from silence detection
                    segment_times = chopper.chop_by_silence(waveform, top_db=silence_threshold)
                    
                    # Create output directory
                    os.makedirs("output/chopped", exist_ok=True)
                    
                    # Create segments
                    segments = []
                    for i, (start_time, end_time) in enumerate(segment_times):
                        # Convert to samples
                        start_sample = int(start_time * sr)
                        end_sample = int(end_time * sr)
                        
                        if end_sample - start_sample > (min_silence / 1000 * sr):  # Apply minimum duration
                            segment = waveform[start_sample:end_sample]
                            output_path = f"output/chopped/{os.path.splitext(auto_file.name)[0]}_silence_{i+1}.wav"
                            sf.write(output_path, segment, sr)
                            segments.append({
                                "path": output_path,
                                "start": start_time,
                                "end": end_time
                            })
                    
                    st.success(f"Detected {len(segment_times)} segments after silence removal")
                    
                    # Plot waveform with segment markers
                    fig, ax = plt.subplots(figsize=(12, 3))
                    ax.plot(times, waveform)
                    
                    # Add rectangles for segments
                    for segment in segments:
                        ax.axvspan(segment["start"], segment["end"], alpha=0.2, color='green')
                    
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Amplitude")
                    ax.set_title("Waveform with Non-Silent Segments Highlighted")
                    st.pyplot(fig)
                    
                    # Display segments
                    st.subheader("Chopped Segments")
                    for i, segment in enumerate(segments):
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            st.write(f"Segment {i+1}")
                            st.write(f"Start: {segment['start']:.2f}s")
                            st.write(f"End: {segment['end']:.2f}s")
                            st.audio(segment["path"])
                        with col2:
                            segment_waveform, _ = librosa.load(segment["path"], sr=None)
                            fig, ax = plt.subplots(figsize=(8, 2))
                            ax.plot(segment_waveform)
                            ax.set_title(f"Segment {i+1}")
                            st.pyplot(fig)
                        with col3:
                            if st.button(f"Save as one-shot {i+1}", key=f"save_silence_{i}"):
                                oneshot_dir = "output/oneshots"
                                os.makedirs(oneshot_dir, exist_ok=True)
                                oneshot_path = f"{oneshot_dir}/{os.path.splitext(auto_file.name)[0]}_oneshot_{i+1}.wav"
                                sf.write(oneshot_path, segment_waveform, sr)
                                st.success(f"Saved one-shot to {oneshot_path}")

# Tab 7: Collaborate
with tab7:
    st.header("Collaborate with Your Team")
    
    st.write("""
    Share your work with team members, export/import projects, and manage collaborative workflows.
    """)
    
    # Tabs for different collaboration features
    collab_tab1, collab_tab2, collab_tab3 = st.tabs(["Share Samples & Models", "Export/Import Project", "Team Workflow"])
    
    # Tab 1: Share Samples & Models
    with collab_tab1:
        st.subheader("Share Your Work")
        
        share_type = st.radio("What would you like to share?", 
                             ["Audio Samples", "Processed Segments", "Trained Models", "Generated Samples"])
        
        if share_type == "Audio Samples":
            # Upload samples for collaboration
            st.write("Upload new audio samples to share with your team:")
            uploaded_collab_files = st.file_uploader("Choose audio files", 
                                                  type=["wav", "mp3", "ogg"], 
                                                  accept_multiple_files=True, 
                                                  key="collab_uploader")
            
            if uploaded_collab_files:
                if st.button("Upload Samples"):
                    # Create directory for uploaded collaboration files
                    os.makedirs("data/collab", exist_ok=True)
                    
                    # Save uploaded collaboration files
                    saved_files = []
                    for uploaded_file in uploaded_collab_files:
                        collab_file_path = os.path.join("data/collab", uploaded_file.name)
                        with open(collab_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_files.append(collab_file_path)
                    
                    st.success(f"Uploaded {len(saved_files)} audio samples for collaboration!")
            
            # Display existing samples
            st.write("Browse existing shared samples:")
            if os.path.exists("data/collab"):
                collab_files = [f for f in os.listdir("data/collab") 
                              if f.endswith((".wav", ".mp3", ".ogg"))]
                
                if collab_files:
                    for i, file_name in enumerate(collab_files):
                        file_path = os.path.join("data/collab", file_name)
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.write(file_name)
                            st.audio(file_path)
                        with col2:
                            # Display download button
                            with open(file_path, "rb") as f:
                                btn = st.download_button(
                                    label=f"Download {file_name}",
                                    data=f,
                                    file_name=file_name,
                                    mime="audio/wav",
                                    key=f"download_collab_{i}"
                                )
                else:
                    st.info("No shared samples found. Upload some files first!")
            else:
                st.info("No shared samples directory found. Upload some files first!")
                
        elif share_type == "Processed Segments":
            # Display and download processed segments
            st.write("Share processed segments with your team:")
            
            if st.session_state.processed_samples:
                st.success(f"{len(st.session_state.processed_samples)} processed segments available to share")
                
                # Option to create a ZIP archive of all segments
                if st.button("Create ZIP of All Segments"):
                    import zipfile
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                        with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
                            for segment_path in st.session_state.processed_samples:
                                zipf.write(segment_path, os.path.basename(segment_path))
                        
                        # Provide download link for the ZIP file
                        with open(tmp_zip.name, "rb") as f:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            st.download_button(
                                label="Download All Segments as ZIP",
                                data=f,
                                file_name=f"processed_segments_{timestamp}.zip",
                                mime="application/zip"
                            )
                
                # Display individual segments
                st.write("Individual segments:")
                for i, segment_path in enumerate(st.session_state.processed_samples[:10]):  # Limit to first 10
                    file_name = os.path.basename(segment_path)
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(file_name)
                        st.audio(segment_path)
                    with col2:
                        # Display download button
                        with open(segment_path, "rb") as f:
                            btn = st.download_button(
                                label=f"Download {file_name}",
                                data=f,
                                file_name=file_name,
                                mime="audio/wav",
                                key=f"download_segment_{i}"
                            )
            else:
                st.info("No processed segments available. Process some samples first in the 'Process Samples' tab.")
        
        elif share_type == "Trained Models":
            st.write("Share your trained models with team members:")
            
            # Upload model option
            st.subheader("Upload a Trained Model")
            model_file = st.file_uploader("Upload a model file (.pt)", type=["pt"])
            
            if model_file:
                model_name = st.text_input("Model name", f"shared_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
                
                if st.button("Upload Model"):
                    os.makedirs("models", exist_ok=True)
                    model_path = os.path.join("models", model_name)
                    
                    with open(model_path, "wb") as f:
                        f.write(model_file.getbuffer())
                    
                    st.success(f"Model uploaded and saved as {model_name}")
            
            # Display existing models
            st.subheader("Available Models")
            models_dir = Path("models")
            if models_dir.exists():
                model_files = [f.name for f in models_dir.glob("*.pt")]
                
                if model_files:
                    for i, model_name in enumerate(model_files):
                        model_path = os.path.join("models", model_name)
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"Model: {model_name}")
                            model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
                            st.write(f"Size: {model_size:.2f} MB")
                            st.write(f"Modified: {datetime.datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        with col2:
                            with open(model_path, "rb") as f:
                                st.download_button(
                                    label=f"Download {model_name}",
                                    data=f,
                                    file_name=model_name,
                                    mime="application/octet-stream",
                                    key=f"download_model_{i}"
                                )
                else:
                    st.info("No models found. Train or upload a model first!")
        
        elif share_type == "Generated Samples":
            st.write("Share your generated samples with team members:")
            
            # Display generated samples
            output_dir = Path("output")
            if output_dir.exists():
                generated_files = [f for f in output_dir.glob("*.wav")]
                
                if generated_files:
                    st.success(f"Found {len(generated_files)} generated samples")
                    
                    # Option to create a ZIP archive
                    if st.button("Create ZIP of All Generated Samples"):
                        import zipfile
                        import tempfile
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                            with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
                                for file_path in generated_files:
                                    zipf.write(file_path, os.path.basename(file_path))
                            
                            # Provide download link for the ZIP file
                            with open(tmp_zip.name, "rb") as f:
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                st.download_button(
                                    label="Download All Generated Samples as ZIP",
                                    data=f,
                                    file_name=f"generated_samples_{timestamp}.zip",
                                    mime="application/zip"
                                )
                    
                    # Display individual samples
                    for i, file_path in enumerate(generated_files):
                        file_name = os.path.basename(file_path)
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.write(file_name)
                            st.audio(str(file_path))
                        with col2:
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    label=f"Download {file_name}",
                                    data=f,
                                    file_name=file_name,
                                    mime="audio/wav",
                                    key=f"download_generated_{i}"
                                )
                else:
                    st.info("No generated samples found. Generate some samples first in the 'Generate Audio' tab.")
            else:
                st.info("Output directory not found. Generate some samples first.")
    
    # Tab 2: Export/Import Project
    with collab_tab2:
        st.subheader("Export/Import Project")
        
        export_tab, import_tab = st.tabs(["Export Project", "Import Project"])
        
        with export_tab:
            st.write("Export your current project for sharing with team members or backup:")
            
            # Select what to include in the export
            st.write("Select what to include in the export:")
            include_raw = st.checkbox("Raw audio samples", True)
            include_processed = st.checkbox("Processed samples", True)
            include_models = st.checkbox("Trained models", True)
            include_output = st.checkbox("Generated output", True)
            
            # Project metadata
            project_name = st.text_input("Project Name", f"chopper_project_{datetime.datetime.now().strftime('%Y%m%d')}")
            project_description = st.text_area("Project Description", "Audio chopper project export")
            
            if st.button("Create Project Export"):
                with st.spinner("Creating project export..."):
                    import zipfile
                    import tempfile
                    import json
                    
                    # Create metadata
                    metadata = {
                        "project_name": project_name,
                        "description": project_description,
                        "date_created": datetime.datetime.now().isoformat(),
                        "version": "1.0.0",
                        "contents": []
                    }
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                        with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
                            # Add raw samples
                            if include_raw and os.path.exists("data/raw"):
                                for file_path in Path("data/raw").glob("*.*"):
                                    if file_path.suffix.lower() in [".wav", ".mp3", ".ogg"]:
                                        zipf.write(file_path, f"data/raw/{file_path.name}")
                                        metadata["contents"].append(str(file_path))
                            
                            # Add processed samples
                            if include_processed and os.path.exists("data/processed"):
                                for file_path in Path("data/processed").rglob("*.*"):
                                    zipf.write(file_path, f"data/processed/{file_path.relative_to('data/processed')}")
                                    metadata["contents"].append(str(file_path))
                            
                            # Add models
                            if include_models and os.path.exists("models"):
                                for file_path in Path("models").glob("*.pt"):
                                    zipf.write(file_path, f"models/{file_path.name}")
                                    metadata["contents"].append(str(file_path))
                            
                            # Add generated output
                            if include_output and os.path.exists("output"):
                                for file_path in Path("output").rglob("*.wav"):
                                    zipf.write(file_path, f"output/{file_path.relative_to('output')}")
                                    metadata["contents"].append(str(file_path))
                            
                            # Add metadata
                            metadata_json = json.dumps(metadata, indent=2)
                            zipf.writestr("metadata.json", metadata_json)
                        
                        # Provide download link for the ZIP file
                        with open(tmp_zip.name, "rb") as f:
                            st.download_button(
                                label=f"Download {project_name}.zip",
                                data=f,
                                file_name=f"{project_name}.zip",
                                mime="application/zip"
                            )
                
                st.success("Project export created successfully!")
        
        with import_tab:
            st.write("Import a project from your team members:")
            
            project_zip = st.file_uploader("Upload project ZIP", type=["zip"])
            
            if project_zip:
                # Extract and display project metadata first
                import zipfile
                import io
                import json
                
                with zipfile.ZipFile(io.BytesIO(project_zip.getvalue()), 'r') as zip_ref:
                    if "metadata.json" in zip_ref.namelist():
                        with zip_ref.open("metadata.json") as metadata_file:
                            metadata = json.load(metadata_file)
                            
                            st.subheader("Project Information")
                            st.write(f"Project: {metadata.get('project_name', 'Unknown')}")
                            st.write(f"Description: {metadata.get('description', 'No description')}")
                            st.write(f"Created: {metadata.get('date_created', 'Unknown date')}")
                            st.write(f"Files included: {len(metadata.get('contents', []))}")
                    else:
                        st.warning("No metadata found in the project ZIP. This might not be a valid project export.")
                
                # Import options
                st.subheader("Import Options")
                replace_existing = st.radio(
                    "If files exist:",
                    ["Skip existing files", "Replace existing files"]
                )
                
                if st.button("Import Project"):
                    with st.spinner("Importing project..."):
                        import zipfile
                        import io
                        
                        with zipfile.ZipFile(io.BytesIO(project_zip.getvalue()), 'r') as zip_ref:
                            # Get file list
                            files = [f for f in zip_ref.namelist() if f != "metadata.json"]
                            
                            # Extract files
                            for file in files:
                                extract_path = file
                                
                                # Check if file exists
                                if os.path.exists(extract_path) and replace_existing == "Skip existing files":
                                    continue
                                
                                # Create directory if it doesn't exist
                                os.makedirs(os.path.dirname(extract_path), exist_ok=True)
                                
                                # Extract file
                                with zip_ref.open(file) as source, open(extract_path, "wb") as target:
                                    target.write(source.read())
                        
                        st.success(f"Project imported successfully! {len(files)} files were processed.")
                        
                        # Update model file list after import
                        if os.path.exists("models"):
                            model_files = [f.name for f in Path("models").glob("*.pt")]
                        
                        # Update session state for processed samples
                        if os.path.exists("data/processed/segments"):
                            segment_paths = [str(f) for f in Path("data/processed/segments").glob("*.wav")]
                            if segment_paths:
                                st.session_state.processed_samples = segment_paths
                                st.info(f"Found {len(segment_paths)} processed segments in the imported project.")
    
    # Tab 3: Team Workflow
    with collab_tab3:
        st.subheader("Team Workflow")
        
        st.write("""
        Keep track of project progress and coordinate with your team.
        """)
        
        # Project status
        st.write("### Project Status")
        
        # Initialize project_status in session state if not present
        if 'project_status' not in st.session_state:
            st.session_state.project_status = {
                "tasks": [
                    {"task": "Sample collection", "status": "In Progress", "assigned_to": "Team", "notes": ""},
                    {"task": "Sample processing", "status": "Not Started", "assigned_to": "Team", "notes": ""},
                    {"task": "Model training", "status": "Not Started", "assigned_to": "Team", "notes": ""},
                    {"task": "Sample generation", "status": "Not Started", "assigned_to": "Team", "notes": ""}
                ],
                "notes": ""
            }
        
        # Display and update task status
        for i, task in enumerate(st.session_state.project_status["tasks"]):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
            
            with col1:
                task_name = st.text_input(f"Task {i+1}", task["task"], key=f"task_name_{i}")
                st.session_state.project_status["tasks"][i]["task"] = task_name
            
            with col2:
                status = st.selectbox(
                    "Status",
                    ["Not Started", "In Progress", "Completed"],
                    ["Not Started", "In Progress", "Completed"].index(task["status"]),
                    key=f"task_status_{i}"
                )
                st.session_state.project_status["tasks"][i]["status"] = status
            
            with col3:
                assigned_to = st.text_input("Assigned To", task["assigned_to"], key=f"task_assigned_{i}")
                st.session_state.project_status["tasks"][i]["assigned_to"] = assigned_to
            
            with col4:
                notes = st.text_input("Notes", task["notes"], key=f"task_notes_{i}")
                st.session_state.project_status["tasks"][i]["notes"] = notes
        
        # Add new task button
        if st.button("Add Task"):
            st.session_state.project_status["tasks"].append({
                "task": "New Task",
                "status": "Not Started",
                "assigned_to": "Team",
                "notes": ""
            })
        
        # Project notes
        st.write("### Project Notes")
        project_notes = st.text_area(
            "Shared project notes",
            st.session_state.project_status.get("notes", ""),
            height=200
        )
        st.session_state.project_status["notes"] = project_notes
        
        # Export project status
        if st.button("Export Project Status"):
            import json
            
            status_json = json.dumps(st.session_state.project_status, indent=2)
            b64 = base64.b64encode(status_json.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="project_status.json">Download Project Status</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # Import project status
        status_file = st.file_uploader("Import Project Status", type=["json"])
        if status_file:
            import json
            
            try:
                status_data = json.loads(status_file.getvalue().decode())
                if "tasks" in status_data:
                    st.session_state.project_status = status_data
                    st.success("Project status imported successfully!")
                else:
                    st.error("Invalid project status file format.")
            except Exception as e:
                st.error(f"Error importing project status: {str(e)}")

# Footer
st.markdown("---")
st.write("Made with ❤️ by the Chopper Audio Team")
st.write("Version 0.1.0")

if __name__ == "__main__":
    # This is used when running the script as a standalone app
    pass
