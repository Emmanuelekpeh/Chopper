import streamlit as st
import os
import librosa
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import zipfile
import tempfile

# Assuming TransformerSampleGenerator and ImprovedSampleGenerator are in core
# and load_model is passed or accessible
# from core.transformer_generator import TransformerSampleGenerator
# from core.improved_generator import ImprovedSampleGenerator

def render_generate_audio_tab(load_model_func, selected_model_type, selected_model_file):
    st.header("Generate New Audio Samples")

    # Model selection and loading
    model_col1, model_col2 = st.columns([2, 1])

    with model_col1:
        st.subheader("Model Selection")

        if selected_model_file == "None":
            st.warning("No model selected in sidebar. Using untrained model.")
        else:
            st.success(f"Model selected: {selected_model_file}")

        if st.button("Load Selected Model"):
            with st.spinner("Loading model..."):
                if selected_model_file == "None":
                    # Use the passed load_model_func and selected_model_type
                    generator = load_model_func(selected_model_type)
                    st.warning("Using untrained model - results may not be usable")
                else:
                    generator = load_model_func(selected_model_type, selected_model_file)
                    st.success(f"Model {selected_model_file} loaded successfully!")

            st.session_state.generator = generator
            st.session_state.model_loaded = True

    with model_col2:
        # Display model info if available
        if st.session_state.get('model_loaded', False): # Use .get for safety
            st.success("Model loaded and ready")
        else:
            st.error("No model loaded")

    # Generation options
    if st.session_state.get('model_loaded', False):
        st.subheader("Generation Options")

        # Add tabs for different generation methods
        gen_tab1, gen_tab2, gen_tab3 = st.tabs(["Basic Generation", "Guided Generation", "Batch Generation"])

        # Tab 1: Basic Generation
        with gen_tab1:
            num_samples = st.slider("Number of samples to generate", 1, 5, 1, key="gen_num_samples")
            sample_length = st.slider("Sample length (seconds)", 1, 10, 5, key="gen_sample_length") # Key added

            if st.button("Generate Sample(s)"):
                with st.spinner("Generating audio samples..."):
                    os.makedirs("output", exist_ok=True)
                    generated_samples_paths = []

                    for i in range(num_samples):
                        output_path = f"output/generated_sample_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.wav"
                        st.session_state.generator.generate_and_save_audio(output_path)
                        generated_samples_paths.append(output_path)

                        waveform, sr = librosa.load(output_path, sr=None)
                        fig, ax = plt.subplots(figsize=(10, 2))
                        ax.plot(waveform)
                        ax.set_title(f"Generated Sample {i+1}")
                        st.pyplot(fig)
                        st.audio(output_path, format="audio/wav")

                        if 'saved_samples' not in st.session_state:
                            st.session_state.saved_samples = []
                        st.session_state.saved_samples.append(output_path)
                    
                    if "project_stats" in st.session_state:
                         st.session_state.project_stats["generated_samples"] += len(generated_samples_paths)


        # Tab 2: Guided Generation
        with gen_tab2:
            st.write("Control the generation process by using a reference sample or adjusting parameters.")
            st.subheader("Reference Track (Optional)")
            reference_file = st.file_uploader("Upload a reference audio file", type=["wav", "mp3", "ogg"], key="reference_uploader")

            if reference_file:
                os.makedirs("data/reference", exist_ok=True)
                ref_path = os.path.join("data/reference", reference_file.name)
                with open(ref_path, "wb") as f:
                    f.write(reference_file.getbuffer())
                st.audio(ref_path)

                if st.button("Generate With Reference"):
                    with st.spinner("Generating RL-optimized sample..."):
                        output_path = f"output/rl_optimized_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                        # Ensure generator has use_rl_optimization
                        if hasattr(st.session_state.generator, 'use_rl_optimization'):
                            audio, _ = st.session_state.generator.use_rl_optimization(target_audio_path=ref_path, n_steps=30)
                            st.session_state.generator.save_audio(audio, output_path)

                            waveform, sr = librosa.load(output_path, sr=None)
                            fig, ax = plt.subplots(figsize=(10, 2))
                            ax.plot(waveform)
                            ax.set_title("RL-Optimized Sample")
                            st.pyplot(fig)
                            st.audio(output_path, format="audio/wav")

                            if 'saved_samples' not in st.session_state:
                                st.session_state.saved_samples = []
                            st.session_state.saved_samples.append(output_path)
                            
                            if "project_stats" in st.session_state:
                                st.session_state.project_stats["generated_samples"] += 1
                        else:
                            st.error("The current model does not support RL optimization.")


        # Tab 3: Batch Generation
        with gen_tab3:
            st.write("Generate multiple samples at once with varying parameters.")
            batch_size = st.slider("Batch size", 5, 50, 10, key="batch_gen_size")

            with st.expander("Advanced Settings"):
                variation = st.slider("Parameter variation (%)", 0, 50, 10, key="batch_gen_variation")
                st.write("Adds random variation to the latent space")

            if st.button("Generate Batch"):
                with st.spinner(f"Generating batch of {batch_size} samples..."):
                    os.makedirs("output/batch", exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                    for i in range(batch_size):
                        output_path = f"output/batch/batch_{timestamp}_{i}.wav"
                        st.session_state.generator.generate_and_save_audio(output_path)

                    st.success(f"Generated {batch_size} samples in output/batch/ directory")

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                        with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
                            batch_files = [f for f in Path(f"output/batch").glob(f"batch_{timestamp}_*.wav")]
                            for file_path in batch_files:
                                zipf.write(file_path, file_path.name)
                        
                        with open(tmp_zip.name, "rb") as f:
                            st.download_button(
                                label=f"Download Batch as ZIP",
                                data=f,
                                file_name=f"batch_samples_{timestamp}.zip",
                                mime="application/zip"
                            )
                    if "project_stats" in st.session_state:
                        st.session_state.project_stats["generated_samples"] += batch_size
        
        st.subheader("Manage Generated Samples")
        if 'saved_samples' in st.session_state and st.session_state.saved_samples:
            st.write(f"You have {len(st.session_state.saved_samples)} saved samples in this session.")
            if st.button("Clear Saved Samples"):
                st.session_state.saved_samples = []
                st.success("Saved samples list cleared")
        else:
            st.info("No samples saved in this session. Generate some samples first!")
    else:
        st.info("Please load a model first using the 'Load Selected Model' button above.") 